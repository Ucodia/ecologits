export class RangeValue {
  constructor(min, max) {
    this.min = min;
    this.max = max;
  }

  get mean() {
    return (this.min + this.max) / 2;
  }

  add(other) {
    if (other instanceof RangeValue) {
      return new RangeValue(this.min + other.min, this.max + other.max);
    }
    return new RangeValue(this.min + other, this.max + other);
  }

  multiply(other) {
    if (other instanceof RangeValue) {
      return new RangeValue(this.min * other.min, this.max * other.max);
    }
    return new RangeValue(this.min * other, this.max * other);
  }

  divide(other) {
    return new RangeValue(this.min / other, this.max / other);
  }
}

export function toRange(value) {
  return value instanceof RangeValue ? value : new RangeValue(value, value);
}
