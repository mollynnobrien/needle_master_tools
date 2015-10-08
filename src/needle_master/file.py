
'''
file utilities
'''

def ParseEnvironmentName(filename):
    toks = filename.split('.')[0].split('_')
    return toks[1]

def ParseDemoName(filename):
    toks = filename.split('.')[0].split('_')
    return (int(toks[1]),toks[2])
