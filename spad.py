from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch
import numpy as np

'''
Creates and writes a SPAD image from the path-tracer annotators.
Args:
    output_path: path to save the SPAD images
    dt: time step for the SPAD sensor
    quantum_efficiency: quantum efficiency of the SPAD sensor
    dark_count_rate: dark count rate of the SPAD sensor
    num_average: number of frames to average over
'''
class SPADWriter(Writer):
    def __init__(self, output_path = "/images", dt = 0.001, quantum_efficiency = 0.5, dark_count_rate = 1., num_average = 20):
        self.dt = dt
        self.quantum_efficiency = quantum_efficiency
        self.dark_count_rate = dark_count_rate
        self.dcrdt = dark_count_rate * dt
        self.num_average = num_average
        self.annotators = []
        self.annotators.append(AnnotatorRegistry.get_annotator("PtGlobalIllumination")) # f16 rgba
        self.annotators.append(AnnotatorRegistry.get_annotator("PtDirectIllumation")) # f16 rgba
        self.annotators.append(AnnotatorRegistry.get_annotator("PtSelfIllumination")) # f16 rgba
        self.annotators.append(AnnotatorRegistry.get_annotator("distance_to_image_plane")) # f32
        self._backend = BackendDispatch({"paths": {"out_dir": output_path}})
        self.time = 0
        self.buffer = []
    
    def write(self, data):
        photon_count = data["PtGlobalIllumination"] + data["PtDirectIllumation"] + data["PtSelfIllumination"]
        photon_count = np.mean(photon_count[:, :, :-1], axis=2, dtype=np.double) # average over RGB channels
        photon_count /= np.square(data["distance_to_image_plane"]) # path-tracer outputs don't account for distance
        spad_image = np.random.rand(*photon_count.shape) > np.exp(-photon_count * self.quantum_efficiency - self.dcrdt) # spad model
        self.buffer.append(spad_image)

        if len(self.buffer) >= self.num_average:
            output = np.mean(self.buffer, axis=0)
            output = (output * 255).astype(np.uint8)
            output = np.stack((output, output, output, np.ones_like(output) * 255), axis=-1)
            self._backend.write_image(f"spad_{self.time}.png", output)
            self.buffer = []
            
        self.time = round(self.time + self.dt, 5)