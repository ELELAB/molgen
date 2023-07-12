import selfies as sf

def selfie_from_tensor(selfie_tensor, index_to_symbol, sos_token_included=True, eos_token="[nop]"):
    """
    Converts a tensor of selfies to a list of selfies.
    """
    n_elements = selfie_tensor.shape[0]
    selfies = []
    for i in range(n_elements):
        if sos_token_included:
            selfie = selfie_tensor[i, 1:]
        else:
            selfie = selfie_tensor[i]
        selfie = [index_to_symbol[str(index.item())] for index in selfie]
        
        # Find index of eos token
        try:
            eos_index = selfie.index(eos_token)
        except ValueError:
            eos_index = len(selfie)

        selfie = selfie[:eos_index]
        selfie = "".join(selfie)
        selfies.append(selfie)
    return selfies

def smiles_from_selfies(selfies):
    """
    Converts a list of selfies to a list of smiles.
    """
    smiles = []
    for selfie in selfies:
        try:
            smile = sf.decoder(selfie)
        except:
            smile = None
        smiles.append(smile)
    return smiles
