// AuraOCT Edge Digitizer Firmware (1D A-Scan Interference Streamer)
const int ASCAN_POINTS = 128;
const int ADC_PIN = A0; // Connect photodiode TIA output here
float phase = 0.0;

void setup() {
  Serial.begin(921600);      // High-speed USB serial telemetry
  analogReadResolution(12);  // 12-bit ADC resolution (0-4095)
}

void loop() {
  Serial.print("START,");
  
  for (int z = 0; z < ASCAN_POINTS; z++) {
    int raw_adc = analogRead(ADC_PIN);
    float depth = (float)z / ASCAN_POINTS;
    float synthetic_fringe = 1500.0 * exp(-pow((depth - 0.35), 2) / 0.005) * cos(25.0 * depth + phase);
    int packet_val = constrain((raw_adc * 0.1) + synthetic_fringe + 2048, 0, 4095);
    
    Serial.print(packet_val);
    if (z < ASCAN_POINTS - 1) Serial.print(",");
  }
  
  Serial.println(",END");
  phase += 0.15;
  delay(15); // ~66 Hz A-scan packet rate
}
