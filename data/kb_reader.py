#coding = utf-8
with open('kb.txt') as f:
    content = f.read().split('\n')
    services = set()
    types = set()
    names = set()
    operations = set()
    for line in content:
        if len(line)>1:
            terms = line.split()
            print len(terms)
            services.add(terms[0])
            types.add(terms[1])
            names.add(terms[2])
            operations.add(terms[3])


with open('slots.txt', 'w') as f:
    s = ' '.join(services)
    t = ' '.join(types)
    n = ' '.join(names)
    o = ' '.join(operations)
    output = '\n'.join([s,t,n,o])
    f.write(output)