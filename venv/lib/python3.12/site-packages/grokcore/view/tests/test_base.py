import doctest
import os
import unittest

from pkg_resources import resource_listdir

import zope.component.eventtesting
from zope.testing import cleanup

import grokcore.view
from grokcore.view.templatereg import file_template_registry


optionflags = (
    doctest.NORMALIZE_WHITESPACE +
    doctest.ELLIPSIS
)


def setUp(test):
    zope.component.eventtesting.setUp(test)
    file_template_registry.ignore_templates('.svn')


def cleanUp(test):
    cleanup.cleanUp()


def suiteFromPackage(name):
    layer_dir = 'base'
    files = resource_listdir(__name__, f'{layer_dir}/{name}')
    suite = unittest.TestSuite()
    for filename in files:
        if filename.endswith('_fixture.py'):
            continue
        if filename == '__init__.py':
            continue
        test = None
        if filename.endswith('.py'):
            dottedname = 'grokcore.view.tests.{}.{}.{}'.format(
                layer_dir, name, filename[:-3])
            test = doctest.DocTestSuite(
                dottedname,
                setUp=setUp,
                tearDown=cleanUp,
                optionflags=optionflags)
        elif filename.endswith('.txt'):
            test = doctest.DocFileSuite(
                os.path.join(layer_dir, name, filename),
                optionflags=optionflags,
                setUp=setUp,
                tearDown=cleanUp,
                globs={'grok': grokcore.view})
        if test is not None:
            suite.addTest(test)
    return suite


def test_suite():
    suite = unittest.TestSuite()
    for name in [
            'contentprovider',
            'directoryresource',
            'skin',
            'template',
            'view']:
        suite.addTest(suiteFromPackage(name))
    suite.addTest(doctest.DocFileSuite(
        '../templatereg.txt',
        optionflags=optionflags,
        setUp=setUp,
        tearDown=cleanUp,
    ))
    return suite
