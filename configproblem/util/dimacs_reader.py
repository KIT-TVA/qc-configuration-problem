import re

class DimacsReader():
    """ Problem reader for very specific problem files of the form
        c 1 A
        c 2 B
        c 3 C
        c attr cost 1 10
        c attr cost 2 2
        c attr cost 3 -4
        p cnf 3 5
        -1 -2 -3 0
        1 -2 3 0
        1 2 -3 0
        1 -2 -3 0
        -1 2 3 0
        Fist comment section lists all literal identifiers + readable name.
        Second comment section lists attributes if avaiablable.
        Afterwards, the usual dimacs content is parsed.
    """
    def __init__(self) -> None:
        self.features = {}
        self.attributeMap = []
        self.nFeatures = 0
        self.nClauses = 0
        self.clauses = []

    def fromFile(self, path):
        idComment = re.compile('c\s*(\d+)\s*(\w+)\s*')
        attrComment = re.compile('c\s*attr\s*(\w+)\s*(\d+)\s*(-?[0-9]\d*(\.\d+)?)\s*')
        otherComment = re.compile('c.*')
        clauses = re.compile('p\s*cnf\s*(\d+)\s*(\d+)')

        f = open(path, "r")
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

        while len(lines) > 0:
            line = lines.pop(0)
            m = idComment.match(line)
            if m:
                id = int(m.group(1))
                name = m.group(2)
                self.features[id] = name
            else:
                m = attrComment.match(line)
                if m:
                    name = m.group(1)
                    id = int(m.group(2))
                    value = float(m.group(3))
                    self.attributeMap.append((name, id, value))
                else:
                    m = clauses.match(line)
                    if m:
                        self.nFeatures = int(m.group(1))
                        self.nClauses = int(m.group(2))
                        self.clauses = [list(map(int, line.split(' '))) for line in lines]
                        break

    def getFeatures(self):
        return self.features.values()

    def getAttributesFromFeature(self, name):
        fid = self.getId(name)
        return [(name,value) for (name, id, value) in self.attributeMap if fid == id]

    def getId(self, name):
        try:
            id = next(key for key, value in self.features.items() if value == name)
            return id
        except StopIteration:
            return -1 # value not contained in features
    
    def getAttributeValuesByName(self, attribute):
        pass

    def toString(self) -> str:
        features = "\n".join(["c {0} {1}".format(key, value) for key,value in self.features.items()])
        attributes = "\n".join(["c attr {0} {1} {2}".format(attr, id, value) for (attr,id,value) in self.attributeMap])
        clauses = '\n'.join([" ".join(map(str,clause)) for clause in self.clauses])
        return '{0}\n{1}\np cnf {2} {3}\n{4}'.format(features, attributes, self.nFeatures, self.nClauses, clauses)

    def __str__(self):
        return self.toString()

# dummy test
# p = DimacsReader()
# p.fromFile("benchmarks\Car.dimacs")
# print(p)
