# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from functools import wraps
import warnings
import re
from locale import getpreferredencoding as _getpreferredencoding

import igorwriter
from igorwriter import builtin_names
from igorwriter.errors import InvalidNameError, RenameWarning

NG_LETTERS = ['"', '\'', ':', ';'] + [chr(i) for i in range(32)]


def _fix_or_raise(fn, on_errors='raise'):
    @wraps(fn)
    def inner(name, *args, **kwargs):
        fixed_name, msg = fn(name, *args, **kwargs)
        if name != fixed_name:
            if on_errors == 'raise':
                raise InvalidNameError(msg)
            else:
                warnings.warn('name %r is fixed as %r (reason: %s)' % (name, fixed_name, msg), RenameWarning)
        return fixed_name
    return inner


def _fix_length(name, max_bytes, encoding=None):
    if encoding is None:
        encoding = _getpreferredencoding()
    bname = name.encode(encoding)
    if len(bname) == 0:
        name = 'wave0'
    if len(bname) > max_bytes:
        while len(name.encode(encoding)) > max_bytes:
            name = name[:-1]
    return name, 'Size of name must be between 1 and %d' % max_bytes


def _fix_ng_letters(name):
    for s in NG_LETTERS:
        name = name.replace(s, '_')
    return name, 'name must not contain " \' : ; or any control characters.'


def _fix_standard(name):
    if not re.match('[a-zA-Z]', name[0]):
        name = 'X' + name
    name = ''.join(s if re.match('[a-zA-Z0-9]', s) else '_' for s in name)
    return name, 'name must start with an alphabet, and must consist of alphanumerics or underscores.'


def _fix_conflicts(name):
    if name.lower() in (builtin_names.operations + builtin_names.functions + builtin_names.keywords + builtin_names.variables):
        name = name + '_'
    return name, 'name must not conflict with built-in operations, functions, etc.'


def check_and_encode(name, liberal=True, long=False, on_errors='raise', encoding=None):
    """

    :param name: name of an object
    :param liberal: whether Liberal Object Names are allowed or not
    :param long: whether Long Object Names (introduced in Igor 8.00) are allowed or not
    :param on_errors: If 'raise', raises InvalidNameError when name is invalid, otherwise tries to fix errors.
    :param encoding: text encoding. If None, it is set with getpreferredencoding()
    :return:
    """
    if encoding is None:
        encoding = _getpreferredencoding()
    MAX_BYTES = 255 if long else 31
    name_before = name
    name_after = None
    while name_before != name_after:
        name_before = name
        name = _fix_or_raise(_fix_length, on_errors=on_errors)(name_before, MAX_BYTES, encoding=encoding)
        name = _fix_or_raise(_fix_ng_letters, on_errors=on_errors)(name)
        if not liberal:
            name = _fix_or_raise(_fix_standard, on_errors=on_errors)(name)
        name = _fix_or_raise(_fix_conflicts, on_errors=on_errors)(name)
        name_after = name
    return name.encode(encoding)
