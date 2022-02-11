
class REFERENCES:
    # ! World Coordinates 
    WC = "WC" 
    # ! Cameras Coordinates 
    CC = "CC"
    # ! World Coordinates only Rot applied
    WC_SO3 = "WC_SO3"
    # ! Room Coordinates 
    ROOM = "ROOM_REF"

class Enum:
    # This label apply only to boundaries
    cam_ref = REFERENCES
    
    
    N0_INITIALIZED = "NO_INITIALIZED"
    INITIALIZED = "INITIALIZED"
 
    LY_OUT_SCALE = "LY_OUT_SCALE" 
    PL_OUT_SCALE = "PL_OUT_SCALE" 
    IN_SCALE = "IN_SCALE" 

    HorizonNet = "HorizonNet"

    # !Corners levels
    CORNER_LV1 = "CORNER_VL1"
    CORNER_LV20 = "CORNER_LV20"
    CORNER_LV21 = "CORNER_LV21"
    CORNER_LV3 = "CORNER_LV3"

    # ! Room Levels
    ROOM_UNLOCKED = "ROOM_UNLOCKED"
    ROOM_LOCKED = "ROOM_LOCKED"


    # ! PATCH MASK THRESHOLD
    PATCH_THR_CONST = "by_const_threshold"
    PATCH_THR_BY_MEAN = "by_dyn_mean"
    PATCH_THR_BY_MED = "by_dyn_med"