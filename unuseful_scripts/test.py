from icecream import ic

a = 'fff'
b = 123
ic(a, b)

ic.disable()
ic.configureOutput(prefix='hello -> ',includeContext=True)
d = {'key': {1: 'one'}}
ic(d['key'][1])

def func():
    return 'func'

ic.enable()
ic(func())
ic.disable()
