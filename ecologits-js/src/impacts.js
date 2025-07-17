import { RangeValue, toRange } from './rangeValue.js';

class BaseImpact {
  constructor(type, name, value, unit) {
    this.type = type;
    this.name = name;
    this.value = toRange(value);
    this.unit = unit;
  }

  add(other) {
    if (other.type !== this.type) {
      throw new Error(`Cannot add ${this.type} with ${other.type}`);
    }
    return new this.constructor(this.value.add(other.value));
  }
}

export class Energy extends BaseImpact {
  constructor(value) {
    super('energy', 'Energy', value, 'kWh');
  }
}

export class GWP extends BaseImpact {
  constructor(value) {
    super('GWP', 'Global Warming Potential', value, 'kgCO2eq');
  }
}

export class ADPe extends BaseImpact {
  constructor(value) {
    super('ADPe', 'Abiotic Depletion Potential (elements)', value, 'kgSbeq');
  }
}

export class PE extends BaseImpact {
  constructor(value) {
    super('PE', 'Primary Energy', value, 'MJ');
  }
}

export class Usage {
  constructor({ energy, gwp, adpe, pe }) {
    this.type = 'usage';
    this.name = 'Usage';
    this.energy = energy;
    this.gwp = gwp;
    this.adpe = adpe;
    this.pe = pe;
  }
}

export class Embodied {
  constructor({ gwp, adpe, pe }) {
    this.type = 'embodied';
    this.name = 'Embodied';
    this.gwp = gwp;
    this.adpe = adpe;
    this.pe = pe;
  }
}

export class Impacts {
  constructor({ energy, gwp, adpe, pe, usage, embodied }) {
    this.energy = energy;
    this.gwp = gwp;
    this.adpe = adpe;
    this.pe = pe;
    this.usage = usage;
    this.embodied = embodied;
  }
}
