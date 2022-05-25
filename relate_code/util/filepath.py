def getRealFilePath(name):
    return "H:/python/bishe/dataset/" + name + "/" + name + "_real.txt"

def getDataFilePath(name):
    with open(r"H:/python/bishe/dataset/karate/karate.gml") as f:
        f.read()

    return "H:/python/bishe/dataset/" + name + "/" + name + ".gml"
def getDataFilePathh(name):
    return "H:/python/bishe/dataset/" + name + "/" + name + ".txt"