import { RangeValue, toRange } from './rangeValue.js';
import { Energy, GWP, ADPe, PE, Usage, Embodied, Impacts } from './impacts.js';

const MODEL_QUANTIZATION_BITS = 4;
const GPU_ENERGY_ALPHA = 8.91e-8;
const GPU_ENERGY_BETA = 1.43e-6;
const GPU_ENERGY_STDEV = 5.19e-7;
const GPU_LATENCY_ALPHA = 8.02e-4;
const GPU_LATENCY_BETA = 2.23e-2;
const GPU_LATENCY_STDEV = 7e-6;
const GPU_MEMORY = 80;
const GPU_EMBODIED_IMPACT_GWP = 143;
const GPU_EMBODIED_IMPACT_ADPE = 5.1e-3;
const GPU_EMBODIED_IMPACT_PE = 1828;
const SERVER_GPUS = 8;
const SERVER_POWER = 1;
const SERVER_EMBODIED_IMPACT_GWP = 3000;
const SERVER_EMBODIED_IMPACT_ADPE = 0.24;
const SERVER_EMBODIED_IMPACT_PE = 38000;
const HARDWARE_LIFESPAN = 5 * 365 * 24 * 60 * 60;
const DATACENTER_PUE = 1.2;

function gpuEnergy(modelActiveParameterCount, outputTokenCount, alpha, beta, stdev) {
  const mean = alpha * modelActiveParameterCount + beta;
  const min = outputTokenCount * (mean - 1.96 * stdev);
  const max = outputTokenCount * (mean + 1.96 * stdev);
  return new RangeValue(Math.max(0, min), max);
}

function generationLatency(modelActiveParameterCount, outputTokenCount, alpha, beta, stdev, requestLatency) {
  const mean = alpha * modelActiveParameterCount + beta;
  const min = outputTokenCount * (mean - 1.96 * stdev);
  const max = outputTokenCount * (mean + 1.96 * stdev);
  const interval = new RangeValue(Math.max(0, min), max);
  if (interval.max < requestLatency) {
    return interval;
  }
  return toRange(requestLatency);
}

function modelRequiredMemory(modelTotalParameterCount, quantizationBits) {
  return 1.2 * modelTotalParameterCount * quantizationBits / 8;
}

function gpuRequiredCount(modelRequiredMemory, gpuMemory) {
  return Math.ceil(modelRequiredMemory / gpuMemory);
}

function serverEnergy(generationLatency, serverPower, serverGpuCount, gpuRequired) {
  const latency = toRange(generationLatency);
  const val = latency.multiply(serverPower).divide(3600).multiply(gpuRequired / serverGpuCount);
  return val;
}

function requestEnergy(pue, serverEnergyVal, gpuRequired, gpuEnergyVal) {
  const energy = serverEnergyVal.add(gpuEnergyVal.multiply(gpuRequired));
  return energy.multiply(pue);
}

function requestUsageImpact(requestEnergyVal, factor) {
  return requestEnergyVal.multiply(factor);
}

function serverGpuEmbodied(gpuRequired, serverGpuCount, serverVal, gpuVal) {
  return (gpuRequired / serverGpuCount) * serverVal + gpuRequired * gpuVal;
}

function requestEmbodied(embodiedVal, serverLifetime, generationLatency) {
  const latency = toRange(generationLatency);
  return latency.divide(serverLifetime).multiply(embodiedVal);
}

function computeLLMImpactsDag(options) {
  const {
    model_active_parameter_count,
    model_total_parameter_count,
    output_token_count,
    request_latency,
    if_electricity_mix_adpe,
    if_electricity_mix_pe,
    if_electricity_mix_gwp,
    model_quantization_bits = MODEL_QUANTIZATION_BITS,
    gpu_energy_alpha = GPU_ENERGY_ALPHA,
    gpu_energy_beta = GPU_ENERGY_BETA,
    gpu_energy_stdev = GPU_ENERGY_STDEV,
    gpu_latency_alpha = GPU_LATENCY_ALPHA,
    gpu_latency_beta = GPU_LATENCY_BETA,
    gpu_latency_stdev = GPU_LATENCY_STDEV,
    gpu_memory = GPU_MEMORY,
    gpu_embodied_gwp = GPU_EMBODIED_IMPACT_GWP,
    gpu_embodied_adpe = GPU_EMBODIED_IMPACT_ADPE,
    gpu_embodied_pe = GPU_EMBODIED_IMPACT_PE,
    server_gpu_count = SERVER_GPUS,
    server_power = SERVER_POWER,
    server_embodied_gwp = SERVER_EMBODIED_IMPACT_GWP,
    server_embodied_adpe = SERVER_EMBODIED_IMPACT_ADPE,
    server_embodied_pe = SERVER_EMBODIED_IMPACT_PE,
    server_lifetime = HARDWARE_LIFESPAN,
    datacenter_pue = DATACENTER_PUE,
  } = options;

  const gpuEnergyVal = gpuEnergy(model_active_parameter_count, output_token_count,
    gpu_energy_alpha, gpu_energy_beta, gpu_energy_stdev);
  const generationLatencyVal = generationLatency(model_active_parameter_count, output_token_count,
    gpu_latency_alpha, gpu_latency_beta, gpu_latency_stdev, request_latency);
  const modelRequiredMem = modelRequiredMemory(model_total_parameter_count, model_quantization_bits);
  const gpuReqCount = gpuRequiredCount(modelRequiredMem, gpu_memory);
  const serverEnergyVal = serverEnergy(generationLatencyVal, server_power, server_gpu_count, gpuReqCount);
  const requestEnergyVal = requestEnergy(datacenter_pue, serverEnergyVal, gpuReqCount, gpuEnergyVal);

  const usageGWP = requestUsageImpact(requestEnergyVal, if_electricity_mix_gwp);
  const usageADPe = requestUsageImpact(requestEnergyVal, if_electricity_mix_adpe);
  const usagePE = requestUsageImpact(requestEnergyVal, if_electricity_mix_pe);

  const serverGpuEmbodiedGwp = serverGpuEmbodied(gpuReqCount, server_gpu_count, server_embodied_gwp, gpu_embodied_gwp);
  const serverGpuEmbodiedAdpe = serverGpuEmbodied(gpuReqCount, server_gpu_count, server_embodied_adpe, gpu_embodied_adpe);
  const serverGpuEmbodiedPe = serverGpuEmbodied(gpuReqCount, server_gpu_count, server_embodied_pe, gpu_embodied_pe);

  const embodiedGwp = requestEmbodied(serverGpuEmbodiedGwp, server_lifetime, generationLatencyVal);
  const embodiedAdpe = requestEmbodied(serverGpuEmbodiedAdpe, server_lifetime, generationLatencyVal);
  const embodiedPe = requestEmbodied(serverGpuEmbodiedPe, server_lifetime, generationLatencyVal);

  return {
    request_energy: requestEnergyVal,
    request_usage_gwp: usageGWP,
    request_usage_adpe: usageADPe,
    request_usage_pe: usagePE,
    request_embodied_gwp: embodiedGwp,
    request_embodied_adpe: embodiedAdpe,
    request_embodied_pe: embodiedPe
  };
}

export function computeLLMImpacts(params) {
  let {
    modelActiveParameterCount,
    modelTotalParameterCount,
    outputTokenCount,
    ifElectricityMixADPe,
    ifElectricityMixPE,
    ifElectricityMixGWP,
    requestLatency = Infinity,
    ...kwargs
  } = params;

  let activeParams = [modelActiveParameterCount];
  let totalParams = [modelTotalParameterCount];

  if (modelActiveParameterCount instanceof RangeValue || modelTotalParameterCount instanceof RangeValue) {
    activeParams = modelActiveParameterCount instanceof RangeValue ?
      [modelActiveParameterCount.min, modelActiveParameterCount.max] :
      [modelActiveParameterCount, modelActiveParameterCount];
    totalParams = modelTotalParameterCount instanceof RangeValue ?
      [modelTotalParameterCount.min, modelTotalParameterCount.max] :
      [modelTotalParameterCount, modelTotalParameterCount];
  }

  const fields = [
    'request_energy', 'request_usage_gwp', 'request_usage_adpe', 'request_usage_pe',
    'request_embodied_gwp', 'request_embodied_adpe', 'request_embodied_pe'
  ];
  const results = {};

  for (let i = 0; i < activeParams.length; i++) {
    const res = computeLLMImpactsDag({
      model_active_parameter_count: activeParams[i],
      model_total_parameter_count: totalParams[i],
      output_token_count: outputTokenCount,
      request_latency: requestLatency,
      if_electricity_mix_adpe: ifElectricityMixADPe,
      if_electricity_mix_pe: ifElectricityMixPE,
      if_electricity_mix_gwp: ifElectricityMixGWP,
      ...kwargs
    });

    for (const field of fields) {
      if (field in results) {
        const minResult = results[field] instanceof RangeValue ? results[field].min : results[field];
        const maxResult = res[field] instanceof RangeValue ? res[field].max : res[field];
        results[field] = new RangeValue(minResult, maxResult);
      } else {
        results[field] = res[field];
      }
    }
  }

  const energy = new Energy(results['request_energy']);
  const gwpUsage = new GWP(results['request_usage_gwp']);
  const adpeUsage = new ADPe(results['request_usage_adpe']);
  const peUsage = new PE(results['request_usage_pe']);
  const gwpEmbodied = new GWP(results['request_embodied_gwp']);
  const adpeEmbodied = new ADPe(results['request_embodied_adpe']);
  const peEmbodied = new PE(results['request_embodied_pe']);

  return new Impacts({
    energy,
    gwp: gwpUsage.add(gwpEmbodied),
    adpe: adpeUsage.add(adpeEmbodied),
    pe: peUsage.add(peEmbodied),
    usage: new Usage({ energy, gwp: gwpUsage, adpe: adpeUsage, pe: peUsage }),
    embodied: new Embodied({ gwp: gwpEmbodied, adpe: adpeEmbodied, pe: peEmbodied })
  });
}
