# ecologits-js

This is a minimal JavaScript port of the [EcoLogits](https://github.com/genai-impact/ecologits) Python library. It implements a subset of the functionality in Node.js so that energy and environmental impacts of LLM inference can be estimated without Python.

The implementation replicates the computation used by `compute_llm_impacts` from the original project.

```
import { computeLLMImpacts } from './src/index.js';

const impacts = computeLLMImpacts({
  modelActiveParameterCount: 7.3,
  modelTotalParameterCount: 7.3,
  outputTokenCount: 200,
  requestLatency: 5,
  ifElectricityMixADPe: 7.37708e-8,
  ifElectricityMixPE: 9.988,
  ifElectricityMixGWP: 0.590478
});

console.log(impacts.energy.value.mean);
```

Only basic calculations are provided as a reference implementation.
