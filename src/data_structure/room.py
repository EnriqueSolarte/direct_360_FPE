from .ocg_patch import OCGPatch

class Room:
    def __init__(self, data_manager):
        self.dt = data_manager
        self.list_ly = []
        self.list_pl = []
        self.is_initialized = False
        self.local_ocg_patch = OCGPatch(data_manager)
        self.list_corners = []
        self.boundary = None
    
    def initialize(self, layout):
        """
        Room class initializer
        """   
        if self.local_ocg_patch.initialize(layout) and not self.is_initialized:
            self.list_ly.append(layout)
            self.is_initialized = True
            
        return self.is_initialized
        