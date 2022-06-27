from torch.hub import load_state_dict_from_url

def download_weights(url):
    weights = load_state_dict_from_url(
        url,
        map_location='cpu', progress=False, check_hash=False
    )
    return weights