class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 2048;
    this.buffer = new Float32Array(this.bufferSize);
    this.index = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    const channel = input[0];
    for (let i = 0; i < channel.length; i += 1) {
      this.buffer[this.index] = channel[i];
      this.index += 1;

      if (this.index === this.bufferSize) {
        this.port.postMessage(this.buffer.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(this.bufferSize);
        this.index = 0;
      }
    }

    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
