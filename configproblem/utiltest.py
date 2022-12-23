from util.hamiltonian_generator import Extended_Modelreader

if __name__ == "__main__":
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    some_model_path = os.path.join(dir_path, "../benchmarks/pc-config.xml")
    reader = Extended_Modelreader()
    reader.readModel(some_model_path)