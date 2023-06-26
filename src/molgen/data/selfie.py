import selfies as sf


class SmileSelfieConverter:
    def __init__(self):
        pass

    def smile_to_selfie(self, smile):
        try:
            selfie = sf.encoder(smile)
        except Exception:
            # print("Error in converting smile to selfie: ", smile)
            selfie = None
        return selfie

    def selfie_to_smile(self, selfie):
        return sf.decoder(selfie)

    def smile_to_selfie_list(self, smiles):
        return [sf.encoder(smile) for smile in smiles]

    def selfie_to_smile_list(self, selfies):
        return [sf.decoder(selfie) for selfie in selfies]

    def add_selfie_to_file(self, dataframe, smile_column="smiles", selfie_column="selfies"):
        dataframe[selfie_column] = dataframe[smile_column].apply(lambda x: self.smile_to_selfie(x))
        return dataframe

    def add_smile_to_file(self, dataframe, smile_column="smiles", selfie_column="selfies"):
        dataframe[smile_column] = dataframe[selfie_column].apply(lambda x: sf.decoder(x))
        return dataframe
