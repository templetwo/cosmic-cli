import doctest
import unittest

from pkg_resources import resource_listdir

import zope.testbrowser.wsgi
from zope.app.wsgi.testlayer import http

import grokcore.traverser


class Layer(
        zope.testbrowser.wsgi.TestBrowserLayer,
        zope.app.wsgi.testlayer.BrowserLayer):
    pass


layer = Layer(grokcore.traverser)


def http_call(method, path, data=None, **kw):
    """Function to help make RESTful calls.

    method - HTTP method to use
    path - testbrowser style path
    data - (body) data to submit
    kw - any request parameters
    """

    if path.startswith('http://localhost'):
        path = path[len('http://localhost'):]
    request_string = f'{method} {path} HTTP/1.1\n'
    for key, value in kw.items():
        request_string += f'{key}: {value}\n'
    if data is not None:
        request_string += '\r\n'
        request_string += data
    return http(request_string, handle_errors=False)


def suiteFromPackage(name):
    layer_dir = 'functional'
    files = resource_listdir(__name__, f'{layer_dir}/{name}')
    suite = unittest.TestSuite()
    for filename in files:
        if not filename.endswith('.py'):
            continue
        if filename == '__init__.py':
            continue

        dottedname = 'grokcore.traverser.tests.{}.{}.{}'.format(
            layer_dir, name, filename[:-3])
        test = doctest.DocTestSuite(
            dottedname,
            extraglobs=dict(http_call=http_call,
                            http=http,
                            getRootFolder=layer.getRootFolder),
            optionflags=(doctest.ELLIPSIS +
                         doctest.NORMALIZE_WHITESPACE +
                         doctest.REPORT_NDIFF +
                         doctest.IGNORE_EXCEPTION_DETAIL))
        test.layer = layer

        suite.addTest(test)
    return suite


def test_suite():
    suite = unittest.TestSuite()
    for name in ['traversal']:
        suite.addTest(suiteFromPackage(name))
    return suite
