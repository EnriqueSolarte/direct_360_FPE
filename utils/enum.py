
class CAM_REF:
    # ! World Coordinates 
    WC = "WC" 
    # ! Cameras Coordinates 
    CC = "CC"
    # ! World Coordinates only Rot applied
    WC_SO3 = "WC_SO3"
    # ! Room Coordinates 
    ROOM = "ROOM_REF"

class ROOM_STATUS:
    OVERLAPPING = "OVERLAPPING"
    READY_FOR_iSPA = "iSPA"
    MERGED = "OVERLAP_DONE"
    FOR_DELETION = "FOR_DELETION"