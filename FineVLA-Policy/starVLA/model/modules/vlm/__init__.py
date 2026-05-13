


def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm

    if "qwen2.5" in vlm_name.lower(): # temp for some ckpt
        from .QWen2_5 import _QWen_VL_Interface 
        return _QWen_VL_Interface(config)
    elif "florence" in vlm_name.lower(): # temp for some ckpt
        from .Florence2 import _Florence_Interface 
        return _Florence_Interface(config)
    elif "qwen3.5" in vlm_name.lower():
        from .QWen3_5 import _QWen3_5_VL_Interface
        return _QWen3_5_VL_Interface(config)
    
    elif "qwen3" in vlm_name.lower():
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)

    else:
        from .QWen3 import _QWen3_VL_Interface
        return _QWen3_VL_Interface(config)
        # raise NotImplementedError(f"VLM model {vlm_name} not implemented")



