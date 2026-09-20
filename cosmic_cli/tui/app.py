"""Pilot Board. Subscribes to LocalMissionBus; paints BoardState."""

from __future__ import annotations

import os
import threading
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Footer, Input, RichLog, Static

from cosmic_cli import __version__, theme
from cosmic_cli.pause_authority import (
    PauseHandle,
    approve_helix_pending,
    approve_pause,
    decline_pause,
)
from cosmic_cli.tui import format as fmt
from cosmic_cli.tui.state import BoardState, Mission, PendingPause, apply_event
from cosmic_cli.tui.widgets import (
    DiffPeek,
    DirectiveBar,
    IdentityBar,
    InstrumentStack,
    MissionRail,
    StepColumn,
)

_TYPED_ACTIONS = frozenset(
    {"quit", "focus_pending", "approve_pause", "decline_pause", "toggle_diff"}
)


def _helix_on() -> bool:
    return "✓" in theme.helix_mark()


def _mission_id(agent: Any) -> Optional[str]:
    mid = getattr(agent, "mission_id", None)
    return mid if isinstance(mid, str) and mid else None


class PilotApp(App):
    """Three-pane mission board. DirectivesUI is the compatibility name."""

    ENABLE_COMMAND_PALETTE = False
    TITLE = "✦ COSMIC CLI"
    SUB_TITLE = "stargazer"

    BINDINGS = [
        Binding("ctrl+k", "prompt_api_key", "api key", priority=True),
        Binding("ctrl+c", "quit", "quit"),
        Binding("q", "quit", "quit"),
        Binding("enter", "deploy_directive", "deploy"),
        Binding("p", "focus_pending", "pending"),
        Binding("y", "approve_pause", "approve", show=False),
        Binding("n", "decline_pause", "decline", show=False),
        Binding("D", "toggle_diff", "diff"),
    ]

    CSS = f"""
    Screen {{ background: {theme.PAGE}; color: {theme.TEXT}; }}
    #identity {{
        height: 1;
        background: {theme.SURFACE};
        color: {theme.MUTED};
        padding: 0 1;
    }}
    #main {{ height: 1fr; }}
    .rail {{
        width: 32;
        background: {theme.PAGE};
        border-right: tall {theme.BORDER};
    }}
    #instruments {{
        border-right: none;
        border-left: tall {theme.BORDER};
        width: 38;
    }}
    .column {{ width: 1fr; }}
    .section-label {{
        color: {theme.MUTED};
        text-style: bold;
        height: 1;
        padding: 0 1;
    }}
    DataTable {{ background: {theme.PAGE}; height: 1fr; }}
    DataTable > .datatable--header {{ color: {theme.MUTED}; text-style: bold; }}
    DataTable > .datatable--cursor {{ background: {theme.CURSOR}; }}
    #step_header {{ height: 1; padding: 0 1; color: {theme.MAGENTA}; }}
    #step_tape {{
        height: 1fr;
        background: {theme.PANEL};
        border: round {theme.BORDER};
        margin: 0 1;
    }}
    #compass_pulse, #pending, #session_meta {{
        height: auto;
        padding: 0 1 1 1;
    }}
    #pending:focus {{
        background: {theme.CURSOR};
    }}
    #diff_peek {{
        height: 12;
        background: {theme.PANEL};
        border-top: tall {theme.BORDER};
        padding: 0 1;
    }}
    #diff_peek.-hidden {{ display: none; }}
    #directive_bar {{
        height: auto;
        padding: 0 1 1 1;
        background: {theme.SURFACE};
    }}
    #directive_input {{ width: 1fr; background: {theme.SURFACE}; border: tall {theme.BORDER}; }}
    #directive_input:focus {{ border: tall {theme.CYAN}; }}
    #deploy_btn {{ background: {theme.BLUE}; color: #ffffff; text-style: bold; min-width: 12; }}
    Footer {{ background: {theme.SURFACE}; }}
    FooterKey {{ background: {theme.SURFACE}; color: {theme.MUTED}; }}
    FooterKey > .footer-key--key {{ background: {theme.CYAN}; color: {theme.PAGE}; }}
    FooterKey > .footer-key--description {{ color: {theme.MUTED}; }}
    """

    def __init__(self, testing: bool = False) -> None:
        super().__init__()
        self.testing = testing
        self.agents: Dict[str, Any] = {}
        self.agents_by_mission: Dict[str, Any] = {}
        self.show_logs: Dict[str, bool] = {}
        self.log_times: Dict[str, list] = {}
        self.figlet = theme.cosmic_figlet("doom")
        self.board = BoardState(helix_on=_helix_on(), floor_ok=None)
        self._unsubs: List[Callable[[], None]] = []
        self._roots: Dict[str, str] = {}
        self._painting = False
        self._commit: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield IdentityBar(self._identity_text())
        with Horizontal(id="main"):
            yield MissionRail(id="missions", classes="rail")
            yield StepColumn(id="steps", classes="column")
            yield InstrumentStack(id="instruments", classes="rail")
        yield DiffPeek(id="diff_peek", classes="-hidden")
        yield DirectiveBar(id="directive_bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#mission_table", DataTable)
        table.add_columns("STATUS", "STEPS", "DIRECTIVE", "BASIS")
        table.cursor_type = "row"
        self._paint()
        if not (
            os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        ) and not self.testing:
            self.action_prompt_api_key()
        if not self.testing:
            self.set_interval(1, self._refresh_panel)
        try:
            self.query_one("#directive_input", Input).focus()
        except Exception:
            pass

    def on_unmount(self) -> None:
        self._detach_bus()

    def _detach_bus(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs.clear()
        for agent in list(self.agents.values()):
            bus = getattr(agent, "_bus", None)
            unsub_fn = getattr(bus, "unsubscribe", None) if bus is not None else None
            if callable(unsub_fn):
                try:
                    unsub_fn(self._on_bus_event)
                except Exception:
                    pass

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in _TYPED_ACTIONS and self._input_is_focused():
            return False
        return True

    def _input_is_focused(self) -> bool:
        try:
            focused = self.focused
        except Exception:
            return False
        if focused is None:
            return False
        if getattr(focused, "id", None) == "directive_input":
            return True
        return isinstance(focused, Input) and getattr(focused, "id", None) != "pending"

    def _q(self, selector: str, cls: type | None = None):
        try:
            if cls is None:
                return self.query_one(selector)
            return self.query_one(selector, cls)
        except Exception:
            return None

    def _board_alive(self) -> bool:
        try:
            return bool(self.is_running) and bool(self.is_mounted)
        except Exception:
            return False

    def action_prompt_api_key(self) -> None:
        from cosmic_cli.ui import APIKeyScreen

        def handle_result(key: str | None) -> None:
            if key:
                self.notify("🔑 API key set for this session.", severity="information")

        self.push_screen(APIKeyScreen(), handle_result)

    def action_deploy_directive(self) -> None:
        self._handle_deploy()

    def action_focus_pending(self) -> None:
        pending = self._q("#pending")
        if pending is not None:
            pending.focus()

    def action_approve_pause(self) -> None:
        self._decide_selected_pause("approved")

    def action_decline_pause(self) -> None:
        self._decide_selected_pause("declined")

    def action_toggle_diff(self) -> None:
        peek = self._q("#diff_peek")
        if peek is None:
            return
        peek.toggle_class("-hidden")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "directive_input":
            self._handle_deploy()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "deploy_btn":
            self._handle_deploy()

    def _handle_deploy(self) -> None:
        input_widget = self._q("#directive_input", Input)
        if input_widget is None:
            return
        directive = input_widget.value.strip()
        if not directive:
            return
        self.add_directive(directive)
        input_widget.value = ""

    def thread_safe_refresh(self, agent=None) -> None:
        try:
            if not self.is_running:
                return
            self.call_from_thread(self._refresh_panel, agent)
        except Exception:
            return

    def add_directive(self, directive) -> None:
        if directive in self.agents:
            self.notify(f"Directive '{directive}' already deployed!", severity="warning")
            return

        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            self.notify(
                "No API key set. Opening prompt (or use Ctrl+K)...",
                severity="warning",
            )
            self.action_prompt_api_key()
            return

        from cosmic_cli.ui import StargazerAgent

        agent = StargazerAgent(
            directive=directive,
            api_key=api_key,
            ui_callback=self.thread_safe_refresh,
        )
        self.agents[directive] = agent
        self.show_logs[directive] = False
        mid = _mission_id(agent)
        if mid:
            self.agents_by_mission[mid] = agent
            root = getattr(agent, "root", None)
            if root is not None:
                self._roots[mid] = str(root)
            self._seed_mission(agent, directive, mid)
        self._subscribe_agent(agent)
        agent.run()
        self._refresh_panel()

    def _subscribe_agent(self, agent: Any) -> None:
        bus = getattr(agent, "_bus", None)
        subscribe = getattr(bus, "subscribe", None) if bus is not None else None
        if not callable(subscribe):
            return
        try:
            unsub = subscribe(self._on_bus_event)
        except Exception:
            return
        if callable(unsub):
            self._unsubs.append(unsub)

    def _seed_mission(self, agent: Any, directive: str, key: str) -> None:
        if key in self.board.missions:
            return
        max_steps = getattr(agent, "max_steps", None)
        try:
            max_steps_i = int(max_steps)
        except (TypeError, ValueError):
            max_steps_i = 30
        mission = Mission(
            key=key,
            directive=directive,
            status=str(getattr(agent, "status", "ready") or "ready"),
            model=str(getattr(agent, "model", "") or ""),
            max_steps=max_steps_i,
            verify_cmd=getattr(agent, "verify_cmd", None),
            session=getattr(agent, "session_id", None)
            if isinstance(getattr(agent, "session_id", None), str)
            else None,
            exec_mode=str(getattr(agent, "exec_mode", "") or "") or None,
        )
        missions = dict(self.board.missions)
        missions[key] = mission
        selected = self.board.selected_key or key
        self.board = replace(self.board, missions=missions, selected_key=selected)

    def _on_bus_event(self, event: dict) -> None:
        try:
            if not getattr(self, "is_running", False):
                return
            # Operator decisions publish synchronously on the UI thread.
            if threading.get_ident() == self._thread_id:
                self.apply_bus_event(event)
            else:
                self.call_from_thread(self.apply_bus_event, event)
        except Exception:
            return

    def apply_bus_event(self, event: dict) -> None:
        try:
            if not isinstance(event, dict):
                return
            self.board = apply_event(self.board, event)
            self._absorb_start_identity(event)
            if event.get("event") == "gate.pause_minted":
                self._on_pause_minted(event)
            if self._board_alive():
                self._paint()
        except Exception:
            return

    def _on_pause_minted(self, rec: dict) -> None:
        try:
            self.notify("PAUSE — operator approval required (token never shown)")
        except Exception:
            pass
        if self.testing:
            return
        pause = self._pause_matching_event(rec)
        if pause is None:
            return
        if not self._should_open_pause_modal(pause):
            return
        self._open_pause_modal(pause)

    def _pause_matching_event(self, rec: dict) -> Optional[PendingPause]:
        sha = rec.get("action_sha256")
        pending_id = rec.get("pending_id")
        mission = rec.get("mission")
        for pause in self.board.pending_pauses:
            if sha and pause.action_sha256 == sha:
                return pause
            if pending_id is not None and pause.pending_id == pending_id:
                return pause
            if mission and pause.mission_key == mission and pause.action_sha256:
                return pause
        return None

    def _should_open_pause_modal(self, pause: PendingPause) -> bool:
        if self.board.selected_key == pause.mission_key:
            return True
        return len(self.board.pending_pauses) == 1

    def _selected_pause(self) -> Optional[PendingPause]:
        selected = self.board.selected_key
        keyed = [p for p in self.board.pending_pauses if p.mission_key == selected]
        if len(keyed) == 1:
            return keyed[0]
        if keyed:
            return keyed[0]
        if len(self.board.pending_pauses) == 1:
            return self.board.pending_pauses[0]
        return None

    def _pause_handle(self, pause: PendingPause) -> PauseHandle:
        pending_id = pause.pending_id
        if pending_id is not None and not isinstance(pending_id, int):
            pending_id = None
        channel = pause.channel if pause.channel in ("local", "helix", "gate") else "local"
        return PauseHandle(
            action_sha256=pause.action_sha256 or "",
            action_summary=pause.action_summary,
            channel=channel,
            mission_id=pause.mission_key,
            pending_id=pending_id,
            expires_at=pause.expires_at,
            rule_id=pause.rule,
        )

    def _open_pause_modal(self, pause: PendingPause) -> None:
        from cosmic_cli.tui.screens.pause import PauseApproveScreen

        handle = self._pause_handle(pause)
        if not handle.action_sha256:
            return

        def _done(choice: str | None) -> None:
            self._apply_pause_choice(choice, handle)

        self.push_screen(PauseApproveScreen(handle), _done)

    def _decide_selected_pause(self, choice: str) -> None:
        pause = self._selected_pause()
        if pause is None:
            return
        self._apply_pause_choice(choice, self._pause_handle(pause))

    def _approval_manager(self, handle: PauseHandle):
        if handle.mission_id:
            agent = self.agents_by_mission.get(handle.mission_id)
            mgr = getattr(agent, "_approval_mgr", None) if agent is not None else None
            if mgr is not None:
                return mgr
        return None

    def _apply_pause_choice(self, choice: str | None, handle: PauseHandle) -> None:
        if choice not in ("approved", "declined"):
            return
        agent = (
            self.agents_by_mission.get(handle.mission_id)
            if handle.mission_id
            else None
        )
        mgr = self._approval_manager(handle)
        kwargs = {"require_tty": not self.testing}
        if mgr is not None:
            kwargs["manager"] = mgr
        if choice == "approved" and handle.channel == "helix":
            helix_tok = ""
            getter = getattr(agent, "helix_pause_token", None) if agent else None
            if callable(getter):
                helix_tok = (
                    getter(
                        pending_id=handle.pending_id,
                        action_sha256=handle.action_sha256,
                    )
                    or ""
                )
            result = approve_helix_pending(
                handle,
                token=helix_tok,
                require_tty=not self.testing,
            )
        elif choice == "approved":
            result = approve_pause(handle, **kwargs)
        else:
            result = decline_pause(handle, **kwargs)
        emit = getattr(agent, "_emit_pause_resolved", None) if agent else None
        if result.outcome == "approved":
            if callable(emit) and handle.action_sha256:
                emit(
                    "approved",
                    handle.action_summary,
                    handle.action_sha256,
                    by="operator",
                    pending_id=handle.pending_id,
                )
            if agent is not None and result.approval_token_id:
                agent.approval_token_id = result.approval_token_id
            try:
                self.notify(
                    "staged one retry — re-run the blocked action "
                    "(same session; consume on retry)"
                )
            except Exception:
                pass
        elif result.outcome == "declined":
            if callable(emit) and handle.action_sha256:
                emit(
                    "declined",
                    handle.action_summary,
                    handle.action_sha256,
                    by="operator",
                    pending_id=handle.pending_id,
                )
            try:
                self.notify("declined — mission stays blocked")
            except Exception:
                pass
        else:
            try:
                self.notify(result.message or result.outcome, severity="error")
            except Exception:
                pass
        if self._board_alive():
            self._paint()

    def _absorb_start_identity(self, rec: dict) -> None:
        name = rec.get("event")
        if name not in ("mission.start", "start"):
            return
        if "helix" in rec:
            self.board = replace(self.board, helix_on=bool(rec.get("helix")))
        key = rec.get("mission")
        root = rec.get("root")
        if isinstance(key, str) and key and root:
            self._roots[key] = str(root)

    def toggle_logs(self, directive) -> None:
        if directive in self.agents:
            self.show_logs[directive] = not self.show_logs[directive]
            self._refresh_panel()
        else:
            self.notify(f"Directive '{directive}' not found!", severity="error")

    def _refresh_panel(self, agent=None) -> None:
        try:
            self._reconcile_from_agents()
            self._paint()
        except Exception:
            pass

    def _reconcile_from_agents(self) -> None:
        missions = dict(self.board.missions)
        changed = False
        for directive, agent in self.agents.items():
            key = _mission_id(agent)
            if key is None:
                for mid, mission in missions.items():
                    if mission.directive == directive:
                        key = mid
                        break
            if not key or key not in missions:
                continue
            mission = missions[key]
            status = str(getattr(agent, "status", mission.status) or mission.status)
            taken = getattr(agent, "steps_taken", mission.steps_taken)
            try:
                taken_i = int(taken)
            except (TypeError, ValueError):
                taken_i = mission.steps_taken
            if status != mission.status or taken_i != mission.steps_taken:
                missions[key] = replace(
                    mission, status=status, steps_taken=taken_i
                )
                changed = True
        if changed:
            self.board = replace(self.board, missions=missions)

    def _selected(self) -> Optional[Mission]:
        key = self.board.selected_key
        if not key:
            return None
        return self.board.missions.get(key)

    def _commit_short(self) -> str:
        if self._commit is None:
            try:
                from cosmic_cli.buildinfo import _provenance_commit

                self._commit = (_provenance_commit() or "")[:7]
            except Exception:
                self._commit = ""
        return self._commit

    def _identity_text(self) -> str:
        selected = self._selected()
        model = ""
        if selected and selected.model:
            model = selected.model
        else:
            for mission in self.board.missions.values():
                if mission.model:
                    model = mission.model
                    break
        if not model:
            try:
                from cosmic_cli.agents import DEFAULT_MODEL

                model = DEFAULT_MODEL
            except Exception:
                model = ""
        return fmt.identity_line(
            version=__version__,
            commit=self._commit_short(),
            model=model,
            helix_on=bool(self.board.helix_on),
            floor_ok=self.board.floor_ok,
            goal=self.board.goal,
        )

    def _paint(self) -> None:
        if not self._board_alive():
            return
        self._painting = True
        try:
            self._paint_identity()
            self._paint_table()
            self._paint_tape()
            self._paint_instruments()
            self._paint_diff()
        except Exception:
            pass
        finally:
            self._painting = False

    def _paint_identity(self) -> None:
        bar = self._q("#identity", Static)
        if bar is None:
            return
        bar.update(self._identity_text())

    def _paint_table(self) -> None:
        table = self._q("#mission_table", DataTable)
        if table is None:
            return
        selected = self.board.selected_key
        table.clear()
        cursor = 0
        for i, (key, mission) in enumerate(self.board.missions.items()):
            status = mission.status or "ready"
            bar = theme.step_bar(mission.steps_taken, mission.max_steps, status)
            directive = fmt.trunc(mission.directive, 22)
            basis = mission.finish_basis or "—"
            table.add_row(
                Text.from_markup(theme.status_markup(status)),
                Text.from_markup(bar),
                directive,
                basis,
                key=key,
            )
            if key == selected:
                cursor = i
        if self.board.missions:
            table.move_cursor(row=cursor, animate=False, scroll=False)

    def _paint_tape(self) -> None:
        header = self._q("#step_header", Static)
        tape = self._q("#step_tape", RichLog)
        if tape is None:
            return
        tape.clear()
        selected = self._selected()
        if header is not None:
            if selected is None:
                header.update(f"[{theme.MUTED}]STEPS[/]")
            else:
                header.update(
                    f"[{theme.MAGENTA}]{fmt.escape_markup(fmt.trunc(selected.directive, 48))}[/]"
                )
        if selected is None:
            return
        for step in selected.steps[-200:]:
            tape.write(fmt.step_line(step))

    def _paint_instruments(self) -> None:
        compass = self._q("#compass_pulse", Static)
        pending = self._q("#pending", Static)
        meta = self._q("#session_meta", Static)
        if compass is not None:
            compass.update(fmt.compass_line(self.board.compass_today))
        if pending is not None:
            pending.update(fmt.pending_line(self.board.pending_pauses))
        selected = self._selected()
        cwd = None
        verify = self.board.verify_cmd_default
        mode = None
        session = None
        if selected is not None:
            cwd = self._roots.get(selected.key)
            agent = self.agents_by_mission.get(selected.key)
            if cwd is None and agent is not None:
                root = getattr(agent, "root", None)
                if root is not None:
                    cwd = str(root)
            verify = selected.verify_cmd or verify
            mode = selected.exec_mode
            session = selected.session
        if meta is not None:
            meta.update(
                fmt.session_meta_line(
                    cwd=cwd,
                    verify_cmd=verify,
                    mode=mode,
                    session=session,
                )
            )

    def _paint_diff(self) -> None:
        body = self._q("#diff_body", RichLog)
        header = self._q("#diff_header", Static)
        selected = self._selected()
        path = selected.last_mutation_path if selected else None
        diff = selected.last_diff if selected else None
        checkpoint = selected.last_checkpoint if selected else None
        if header is not None:
            label = "DIFF"
            if path:
                label = f"DIFF  {fmt.trunc(path, 48)}"
            header.update(label)
        if body is None:
            return
        body.clear()
        body.write(fmt.diff_body(path, diff, checkpoint))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if self._painting:
            return
        if getattr(event.data_table, "id", None) != "mission_table":
            return
        key = getattr(event.row_key, "value", None)
        if not isinstance(key, str) or not key:
            return
        if key == self.board.selected_key:
            return
        if key not in self.board.missions:
            return
        self.board = replace(self.board, selected_key=key)
        try:
            self._paint_identity()
            self._paint_tape()
            self._paint_instruments()
            self._paint_diff()
        except Exception:
            pass
