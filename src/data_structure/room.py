from .ocg_patch import OCGPatches

class Room:
    def __init__(self, data_manager):
        self.dt = data_manager
        self.list_ly = []
        self.list_pl = []
        self.is_initialized = False
        self.local_ocg_patches = OCGPatches(self.dt)
        self.list_corners = []
        self.boundary = None
        
        # ! For Tracking pose likelihood
        self.p_pose = []
    
    def initialize(self, layout):
        """
        Room class initializer
        """   
        self.is_initialized = False
        if not layout.is_initialized:
            raise ValueError("Layout must be initialized to init a Room instance...")
        
        if not self.local_ocg_patches.initialize(layout.patch):
            return self.is_initialized
        
        self.list_ly.append(layout)
        [self.list_pl.append(pl) for pl in layout.list_pl]
        
        self.is_initialized = True
        
        return self.is_initialized
    
    def add_layout(self, layout):
        self.list_ly.append(layout)
        [self.list_pl.append(pl) for pl in layout.list_pl]
        self.local_ocg_patches.add_patch(layout.patch)