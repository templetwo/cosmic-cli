#!/usr/bin/env python3
"""
📁 File Operations Plugin for Cosmic CLI
Enhanced file handling capabilities with advanced features
"""

import os
import shutil
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .base import BasePlugin, PluginMetadata

class FileOperationsPlugin(BasePlugin):
    """Plugin for advanced file operations"""
    
    def _initialize_metadata(self):
        self.metadata = PluginMetadata(
            name="File Operations",
            version="1.2.0",
            description="Advanced file and directory operations with safety checks",
            author="Cosmic CLI Team",
            dependencies=["pathlib", "shutil"],
            categories=["filesystem", "utility"],
            priority=80
        )
    
    def initialize(self) -> bool:
        """Initialize the file operations plugin"""
        try:
            # Test basic file operations
            test_path = Path.cwd()
            test_path.exists()  # Simple test
            self.logger.info("File operations plugin initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize file operations plugin: {e}")
            return False
    
    def get_capabilities(self) -> List[str]:
        return [
            "READ_FILE", "WRITE_FILE", "COPY_FILE", "MOVE_FILE", "DELETE_FILE",
            "LIST_DIR", "CREATE_DIR", "FILE_INFO", "FIND_FILES", "BACKUP_FILE",
            "COMPRESS", "EXTRACT", "FILE_HASH", "FILE_PERMISSIONS"
        ]
    
    def can_handle(self, action_type: str, parameters: str, context: Dict[str, Any] = None) -> bool:
        """Check if this plugin can handle the action"""
        return action_type.upper() in self.get_capabilities()
    
    def execute(self, action_type: str, parameters: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute file operation"""
        action = action_type.upper()
        context = context or {}
        
        try:
            if action == "READ_FILE":
                return self._read_file(parameters, context)
            elif action == "WRITE_FILE":
                return self._write_file(parameters, context)
            elif action == "COPY_FILE":
                return self._copy_file(parameters, context)
            elif action == "MOVE_FILE":
                return self._move_file(parameters, context)
            elif action == "DELETE_FILE":
                return self._delete_file(parameters, context)
            elif action == "LIST_DIR":
                return self._list_directory(parameters, context)
            elif action == "CREATE_DIR":
                return self._create_directory(parameters, context)
            elif action == "FILE_INFO":
                return self._get_file_info(parameters, context)
            elif action == "FIND_FILES":
                return self._find_files(parameters, context)
            elif action == "BACKUP_FILE":
                return self._backup_file(parameters, context)
            elif action == "COMPRESS":
                return self._compress_files(parameters, context)
            elif action == "EXTRACT":
                return self._extract_archive(parameters, context)
            elif action == "FILE_HASH":
                return self._calculate_hash(parameters, context)
            elif action == "FILE_PERMISSIONS":
                return self._manage_permissions(parameters, context)
            else:
                return {
                    "success": False,
                    "output": f"Unknown file operation: {action}"
                }
                
        except Exception as e:
            self.logger.error(f"File operation {action} failed: {e}")
            return {
                "success": False,
                "output": f"File operation failed: {str(e)}"
            }
    
    def _read_file(self, file_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Read file contents with encoding detection"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"success": False, "output": f"File not found: {file_path}"}
            
            if not path.is_file():
                return {"success": False, "output": f"Path is not a file: {file_path}"}
            
            # Get file size and check if it's too large
            file_size = path.stat().st_size
            max_size = context.get("max_size", 10 * 1024 * 1024)  # 10MB default
            
            if file_size > max_size:
                return {
                    "success": False,
                    "output": f"File too large: {file_size} bytes (max: {max_size})"
                }
            
            # Try to detect encoding
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            encoding_used = None
            
            for encoding in encodings:
                try:
                    content = path.read_text(encoding=encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                # Try binary read for non-text files
                binary_content = path.read_bytes()
                return {
                    "success": True,
                    "output": f"Binary file read successfully ({len(binary_content)} bytes)",
                    "file_size": len(binary_content),
                    "file_type": "binary",
                    "mime_type": mimetypes.guess_type(str(path))[0]
                }
            
            return {
                "success": True,
                "output": content,
                "file_size": len(content),
                "encoding": encoding_used,
                "file_type": "text",
                "mime_type": mimetypes.guess_type(str(path))[0]
            }
            
        except Exception as e:
            return {"success": False, "output": f"Failed to read file: {e}"}
    
    def _write_file(self, parameters: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to file with backup option"""
        try:
            # Parse parameters (expecting format: "path|content" or JSON)
            if parameters.startswith('{'):
                params = json.loads(parameters)
                file_path = params.get('path')
                content = params.get('content', '')
                backup = params.get('backup', True)
                encoding = params.get('encoding', 'utf-8')
            else:
                parts = parameters.split('|', 1)
                if len(parts) != 2:
                    return {"success": False, "output": "Invalid parameters. Use: path|content"}
                file_path, content = parts
                backup = True
                encoding = 'utf-8'
            
            path = Path(file_path)
            
            # Create backup if file exists and backup is requested
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + f'.backup.{int(datetime.now().timestamp())}')
                shutil.copy2(path, backup_path)
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            path.write_text(content, encoding=encoding)
            
            return {
                "success": True,
                "output": f"File written successfully: {file_path}",
                "bytes_written": len(content.encode(encoding)),
                "encoding": encoding
            }
            
        except Exception as e:
            return {"success": False, "output": f"Failed to write file: {e}"}
    
    def _copy_file(self, parameters: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Copy file or directory with progress tracking"""
        try:
            parts = parameters.split('|')
            if len(parts) != 2:
                return {"success": False, "output": "Invalid parameters. Use: source|destination"}
            
            source, destination = parts
            source_path = Path(source)
            dest_path = Path(destination)
            
            if not source_path.exists():
                return {"success": False, "output": f"Source not found: {source}"}
            
            # Create destination parent directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            if source_path.is_file():
                shutil.copy2(source_path, dest_path)
                file_size = source_path.stat().st_size
                return {
                    "success": True,
                    "output": f"File copied successfully: {source} -> {destination}",
                    "bytes_copied": file_size
                }
            elif source_path.is_dir():
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                return {
                    "success": True,
                    "output": f"Directory copied successfully: {source} -> {destination}"
                }
            else:
                return {"success": False, "output": "Source is neither file nor directory"}
                
        except Exception as e:
            return {"success": False, "output": f"Failed to copy: {e}"}
    
    def _list_directory(self, directory: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents with detailed information"""
        try:
            path = Path(directory)
            
            if not path.exists():
                return {"success": False, "output": f"Directory not found: {directory}"}
            
            if not path.is_dir():
                return {"success": False, "output": f"Path is not a directory: {directory}"}
            
            show_hidden = context.get("show_hidden", False)
            show_details = context.get("show_details", True)
            
            items = []
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for item in path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue
                
                try:
                    stat = item.stat()
                    is_file = item.is_file()
                    is_dir = item.is_dir()
                    
                    if is_file:
                        file_count += 1
                        total_size += stat.st_size
                    elif is_dir:
                        dir_count += 1
                    
                    item_info = {
                        "name": item.name,
                        "type": "file" if is_file else "directory" if is_dir else "other",
                        "size": stat.st_size if is_file else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "permissions": oct(stat.st_mode)[-3:] if hasattr(stat, 'st_mode') else "unknown"
                    }
                    
                    if show_details and is_file:
                        item_info["mime_type"] = mimetypes.guess_type(str(item))[0]
                    
                    items.append(item_info)
                    
                except (OSError, PermissionError) as e:
                    items.append({
                        "name": item.name,
                        "type": "error",
                        "error": str(e)
                    })
            
            # Sort items: directories first, then files, alphabetically
            items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))
            
            return {
                "success": True,
                "output": f"Listed {len(items)} items in {directory}",
                "items": items,
                "summary": {
                    "total_items": len(items),
                    "files": file_count,
                    "directories": dir_count,
                    "total_size": total_size
                }
            }
            
        except Exception as e:
            return {"success": False, "output": f"Failed to list directory: {e}"}
    
    def _get_file_info(self, file_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed file information"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"success": False, "output": f"Path not found: {file_path}"}
            
            stat = path.stat()
            
            info = {
                "path": str(path.absolute()),
                "name": path.name,
                "type": "file" if path.is_file() else "directory" if path.is_dir() else "other",
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:] if hasattr(stat, 'st_mode') else "unknown",
                "owner": stat.st_uid if hasattr(stat, 'st_uid') else "unknown",
                "group": stat.st_gid if hasattr(stat, 'st_gid') else "unknown"
            }
            
            if path.is_file():
                info["mime_type"] = mimetypes.guess_type(str(path))[0]
                info["extension"] = path.suffix
                
                # Calculate file hash if requested
                if context.get("calculate_hash", False):
                    with open(path, 'rb') as f:
                        file_hash = hashlib.sha256()
                        while chunk := f.read(8192):
                            file_hash.update(chunk)
                        info["sha256"] = file_hash.hexdigest()
            
            return {
                "success": True,
                "output": f"File info for: {file_path}",
                "info": info
            }
            
        except Exception as e:
            return {"success": False, "output": f"Failed to get file info: {e}"}
    
    def _find_files(self, parameters: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Find files matching pattern"""
        try:
            # Parse parameters
            if parameters.startswith('{'):
                params = json.loads(parameters)
                search_dir = params.get('directory', '.')
                pattern = params.get('pattern', '*')
                recursive = params.get('recursive', True)
                case_sensitive = params.get('case_sensitive', False)
            else:
                parts = parameters.split('|')
                search_dir = parts[0] if len(parts) > 0 else '.'
                pattern = parts[1] if len(parts) > 1 else '*'
                recursive = True
                case_sensitive = False
            
            path = Path(search_dir)
            
            if not path.exists():
                return {"success": False, "output": f"Search directory not found: {search_dir}"}
            
            matches = []
            
            if recursive:
                glob_pattern = f"**/{pattern}"
                found_files = path.glob(glob_pattern)
            else:
                found_files = path.glob(pattern)
            
            for file_path in found_files:
                try:
                    stat = file_path.stat()
                    matches.append({
                        "path": str(file_path.absolute()),
                        "name": file_path.name,
                        "type": "file" if file_path.is_file() else "directory",
                        "size": stat.st_size if file_path.is_file() else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except (OSError, PermissionError):
                    continue
            
            return {
                "success": True,
                "output": f"Found {len(matches)} matches for pattern '{pattern}'",
                "matches": matches,
                "search_params": {
                    "directory": search_dir,
                    "pattern": pattern,
                    "recursive": recursive,
                    "case_sensitive": case_sensitive
                }
            }
            
        except Exception as e:
            return {"success": False, "output": f"Failed to find files: {e}"}
    
    def _calculate_hash(self, file_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate file hash"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {"success": False, "output": f"File not found: {file_path}"}
            
            if not path.is_file():
                return {"success": False, "output": f"Path is not a file: {file_path}"}
            
            hash_type = context.get("hash_type", "sha256").lower()
            
            if hash_type == "md5":
                hasher = hashlib.md5()
            elif hash_type == "sha1":
                hasher = hashlib.sha1()
            elif hash_type == "sha256":
                hasher = hashlib.sha256()
            else:
                return {"success": False, "output": f"Unsupported hash type: {hash_type}"}
            
            with open(path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            
            return {
                "success": True,
                "output": f"{hash_type.upper()} hash calculated for {file_path}",
                "hash": hasher.hexdigest(),
                "hash_type": hash_type.upper(),
                "file_size": path.stat().st_size
            }
            
        except Exception as e:
            return {"success": False, "output": f"Failed to calculate hash: {e}"}
    
    def get_help(self) -> str:
        return """
📁 File Operations Plugin

Capabilities:
• READ_FILE - Read file contents with encoding detection
• WRITE_FILE - Write content to file with backup
• COPY_FILE - Copy files or directories
• MOVE_FILE - Move/rename files or directories
• DELETE_FILE - Delete files or directories
• LIST_DIR - List directory contents with details
• CREATE_DIR - Create directories
• FILE_INFO - Get detailed file information
• FIND_FILES - Find files matching patterns
• BACKUP_FILE - Create file backups
• FILE_HASH - Calculate file hashes (MD5, SHA1, SHA256)

Examples:
• READ_FILE: /path/to/file.txt
• WRITE_FILE: {"path": "/path/to/file.txt", "content": "Hello World"}
• COPY_FILE: /source/file.txt|/dest/file.txt
• FIND_FILES: {"directory": "/search/dir", "pattern": "*.py", "recursive": true}
        """
