import dataclasses
from WorldBuilders.Types import Sampler_T, Layer_T


@dataclasses.dataclass
class RockGenerationConfs:
    enable: bool = dataclasses.field(default_factory=bool)
    collection: list = dataclasses.field(default_factory=list)
    sampler: Sampler_T = Sampler_T()
    layer: Layer_T = Layer_T()