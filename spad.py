import numpy as np
from omni.replicator.core import settings, Writer, AnnotatorRegistry, BackendDispatch

settings.carb_settings(setting="rtx-transient.aov.enableRtxAovs", value=True) # enables path-tracer annotators
settings.carb_settings(setting="rtx-transient.aov.enableRtxAovsSecondary", value=True)

class SPADWriter(Writer):
    '''
    Creates and writes a SPAD image from the path-tracer annotators.
    It computes an image on every step of the simulation and saves averages over a number of frames.

    Args:
        - output_path: path to save the SPAD images
        - dt: time step for the SPAD sensor
        - quantum_efficiency: quantum efficiency of the SPAD sensor
        - dark_count_rate: dark count rate of the SPAD sensor
        - pixel_area: pixel area of the SPAD sensor
        - num_average: number of frames to average over
        - log_saves: whether to print the saved image filenames
    '''
    def __init__(
        self,
        output_path: str = "/images",
        dt: float = 1e-5,
        quantum_efficiency: float = 0.5,
        dark_count_rate: float = 0.01,
        pixel_area: float = 1e-12,
        num_average: int = 20,
        log_saves: bool = True
    ):
        self.dt = dt
        self.quantum_efficiency = quantum_efficiency
        self.dark_count_rate = dark_count_rate
        self.pixel_area = pixel_area
        self.num_average = num_average
        self.log_saves = log_saves
        self.annotators = []
        self.annotators.append(AnnotatorRegistry.get_annotator("PtGlobalIllumination")) # f16 rgba
        self.annotators.append(AnnotatorRegistry.get_annotator("PtDirectIllumation")) # f16 rgba
        self.annotators.append(AnnotatorRegistry.get_annotator("PtSelfIllumination")) # f16 rgba
        self._backend = BackendDispatch({"paths": {"out_dir": output_path}})
        self._frame_id = 0
        self.time = 0
        self.buffer = []
        self.last_frame = None
    
    def write(self, data):
        illuminance = data["PtGlobalIllumination"] + data["PtDirectIllumation"] + data["PtSelfIllumination"]
        photon_count = self.pixel_area * 1e19 * np.average(illuminance[:, :, :-1], weights=[1./3.153, 1./3.734, 1./4.272], axis=2) # illuminance to photon count
        spad_image = np.random.rand(*photon_count.shape) > np.exp(-self.quantum_efficiency * self.dt * photon_count - self.dark_count_rate * self.dt) # spad model
        self.buffer.append(spad_image)

        if len(self.buffer) >= self.num_average:
            output = np.mean(self.buffer, axis=0)
            output = (output * 255).astype(np.uint8)
            output = np.stack((output, output, output, np.ones_like(output) * 255), axis=-1)
            filename = f"spad_{self._frame_id}_{self.time}.png"
            self._backend.write_image(filename, output)
            self.last_frame = output
            if self.log_saves: print(f"Saved SPAD image: {filename}")
            self.buffer = []

        self._frame_id += 1
        self.time = round(self.time + self.dt, 6)
    
    def get_frame(self) -> np.ndarray:
        '''Returns the last averaged SPAD frame.'''
        if self.last_frame is None: return None
        return self.last_frame.copy()
    
    def get_subframe(self) -> np.ndarray:
        '''Returns the last SPAD frame from the averaging buffer.'''
        if len(self.buffer) > 0: return self.buffer[-1].copy()
        return self.get_frame()