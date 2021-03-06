'''
From NLTK decorators: https://github.com/nltk/nltk/blob/develop/nltk/decorators.py

"""
Decorator module by Michele Simionato <michelesimionato@libero.it>
Copyright Michele Simionato, distributed under the terms of the BSD License (see below).
http://www.phyast.pitt.edu/~micheles/python/documentation.html
Included in NLTK for its support of a nice memoization decorator.
"""
'''


import inspect

def __legacysignature(signature):
    """
    For retrocompatibility reasons, we don't use a standard Signature.
    Instead, we use the string generated by this method.
    Basically, from a Signature we create a string and remove the default values.
    """
    listsignature = str(signature)[1:-1].split(",")
    for counter, param in enumerate(listsignature):
        if param.count("=") > 0:
            listsignature[counter] = param[0:param.index("=")].strip()
        else:
            listsignature[counter] = param.strip()
    return ", ".join(listsignature)

def getinfo(func):
    """
    Returns an info dictionary containing:
    - name (the name of the function : str)
    - argnames (the names of the arguments : list)
    - defaults (the values of the default arguments : tuple)
    - signature (the signature : str)
    - fullsignature (the full signature : Signature)
    - doc (the docstring : str)
    - module (the module name : str)
    - dict (the function __dict__ : str)
    >>> def f(self, x=1, y=2, *args, **kw): pass
    >>> info = getinfo(f)
    >>> info["name"]
    'f'
    >>> info["argnames"]
    ['self', 'x', 'y', 'args', 'kw']
    >>> info["defaults"]
    (1, 2)
    >>> info["signature"]
    'self, x, y, *args, **kw'
    >>> info["fullsignature"]
    <Signature (self, x=1, y=2, *args, **kw)>
    """
    assert inspect.ismethod(func) or inspect.isfunction(func)
    argspec = inspect.getfullargspec(func)
    regargs, varargs, varkwargs = argspec[:3]
    argnames = list(regargs)
    if varargs:
        argnames.append(varargs)
    if varkwargs:
        argnames.append(varkwargs)
    fullsignature = inspect.signature(func)
    # Convert Signature to str
    signature = __legacysignature(fullsignature)


    # pypy compatibility
    if hasattr(func, "__closure__"):
        _closure = func.__closure__
        _globals = func.__globals__
    else:
        _closure = func.func_closure
        _globals = func.func_globals

    return dict(
        name=func.__name__,
        argnames=argnames,
        signature=signature,
        fullsignature=fullsignature,
        defaults=func.__defaults__,
        doc=func.__doc__,
        module=func.__module__,
        dict=func.__dict__,
        globals=_globals,
        closure=_closure,
    )

##########################     LEGALESE    ###############################

##   Redistributions of source code must retain the above copyright
##   notice, this list of conditions and the following disclaimer.
##   Redistributions in bytecode form must reproduce the above copyright
##   notice, this list of conditions and the following disclaimer in
##   the documentation and/or other materials provided with the
##   distribution.

##   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
##   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
##   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
##   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
##   HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
##   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
##   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
##   OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
##   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
##   TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
##   USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
##   DAMAGE.
