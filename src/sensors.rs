//! Raspberry Pi GPIO/I2C/SPI/UART sensor drivers for ALICE-Edge
//!
//! Provides traits and drivers for common IoT sensors on Raspberry Pi 5.
//! All drivers produce `&[i32]` slices suitable for `fit_linear_fixed()`.
//!
//! # Supported Sensors
//!
//! | Sensor | Protocol | Measurements |
//! |--------|----------|-------------|
//! | BME280 | I2C | Temperature, Humidity, Pressure |
//! | DHT22 | GPIO | Temperature, Humidity |
//! | ADXL345 | SPI | 3-axis Acceleration |
//! | GPS (NMEA) | UART | Latitude, Longitude, Altitude |
//!
//! # Example
//!
//! ```ignore
//! use alice_edge::sensors::{Bme280Sensor, SensorDriver};
//!
//! let mut bme = Bme280Sensor::new(0x76)?;
//! let samples = bme.read_samples(100, Duration::from_millis(100))?;
//! let (slope, intercept) = fit_linear_fixed(&samples.temperature);
//! ```
//!
//! Author: Moroya Sakamoto

use std::time::{Duration, Instant};

/// Sensor reading with timestamp
#[derive(Debug, Clone)]
pub struct TimestampedReading {
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
    /// Raw sensor value (scaled integer for Q16.16 compatibility)
    pub value: i32,
}

/// Multi-channel sensor data batch
#[derive(Debug, Clone)]
pub struct SensorBatch {
    /// Sensor identifier
    pub sensor_id: &'static str,
    /// Temperature readings (×100, e.g., 2500 = 25.00°C)
    pub temperature: Vec<i32>,
    /// Humidity readings (×100, e.g., 6500 = 65.00%)
    pub humidity: Vec<i32>,
    /// Pressure readings (×10, e.g., 10132 = 1013.2 hPa)
    pub pressure: Vec<i32>,
    /// Acceleration X (×1000, milli-g)
    pub accel_x: Vec<i32>,
    /// Acceleration Y (×1000, milli-g)
    pub accel_y: Vec<i32>,
    /// Acceleration Z (×1000, milli-g)
    pub accel_z: Vec<i32>,
    /// Latitude (×1_000_000, microdegrees)
    pub latitude: Vec<i32>,
    /// Longitude (×1_000_000, microdegrees)
    pub longitude: Vec<i32>,
    /// Altitude (×100, centimeters)
    pub altitude: Vec<i32>,
    /// Sample timestamps (milliseconds since start)
    pub timestamps: Vec<u64>,
}

impl SensorBatch {
    fn new(sensor_id: &'static str) -> Self {
        Self {
            sensor_id,
            temperature: Vec::new(),
            humidity: Vec::new(),
            pressure: Vec::new(),
            accel_x: Vec::new(),
            accel_y: Vec::new(),
            accel_z: Vec::new(),
            latitude: Vec::new(),
            longitude: Vec::new(),
            altitude: Vec::new(),
            timestamps: Vec::new(),
        }
    }
}

/// Trait for all sensor drivers
pub trait SensorDriver {
    /// Initialize the sensor
    fn init(&mut self) -> Result<(), SensorError>;
    /// Read a batch of samples (default interval: 0ms for maximum speed)
    fn read_samples(&mut self, count: usize) -> Result<SensorBatch, SensorError> {
        self.read_samples_interval(count, Duration::from_millis(0))
    }
    /// Read a batch of samples at the given interval
    fn read_samples_interval(
        &mut self,
        count: usize,
        interval: Duration,
    ) -> Result<SensorBatch, SensorError>;
    /// Get sensor name
    fn name(&self) -> &'static str;
}

/// Sensor errors
#[derive(Debug)]
pub enum SensorError {
    /// I2C communication error
    I2c(String),
    /// SPI communication error
    Spi(String),
    /// GPIO error
    Gpio(String),
    /// UART error
    Uart(String),
    /// Sensor not found or not responding
    NotFound(String),
    /// Invalid data received
    InvalidData(String),
    /// Timeout
    Timeout,
}

impl std::fmt::Display for SensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensorError::I2c(e) => write!(f, "I2C error: {}", e),
            SensorError::Spi(e) => write!(f, "SPI error: {}", e),
            SensorError::Gpio(e) => write!(f, "GPIO error: {}", e),
            SensorError::Uart(e) => write!(f, "UART error: {}", e),
            SensorError::NotFound(e) => write!(f, "Sensor not found: {}", e),
            SensorError::InvalidData(e) => write!(f, "Invalid data: {}", e),
            SensorError::Timeout => write!(f, "Sensor timeout"),
        }
    }
}

impl std::error::Error for SensorError {}

// ============================================================
// BME280 — Temperature / Humidity / Pressure (I2C)
// ============================================================

/// BME280 I2C sensor driver
///
/// Bosch BME280: Temperature (-40~85°C), Humidity (0~100%), Pressure (300~1100 hPa)
/// I2C address: 0x76 (default) or 0x77
///
/// # Pi 5 Wiring
///
/// | BME280 | Pi 5 GPIO |
/// |--------|-----------|
/// | VCC | 3.3V (Pin 1) |
/// | GND | GND (Pin 6) |
/// | SDA | GPIO 2 (Pin 3) |
/// | SCL | GPIO 3 (Pin 5) |
#[allow(dead_code)]
pub struct Bme280Sensor {
    address: u16,
    #[cfg(feature = "sensors-hw")]
    i2c: rppal::i2c::I2c,
    // Calibration data (used by compensate_* methods with sensors-hw)
    dig_t1: u16,
    dig_t2: i16,
    dig_t3: i16,
    dig_h1: u8,
    dig_h2: i16,
    dig_h3: u8,
    dig_h4: i16,
    dig_h5: i16,
    dig_h6: i8,
    dig_p1: u16,
    dig_p2: i16,
    dig_p3: i16,
    dig_p4: i16,
    dig_p5: i16,
    dig_p6: i16,
    dig_p7: i16,
    dig_p8: i16,
    dig_p9: i16,
    initialized: bool,
}

impl Bme280Sensor {
    /// Create a new BME280 sensor
    ///
    /// * `bus` - I2C bus number (1 on Pi 5)
    /// * `address` - I2C address (0x76 default, 0x77 alternate)
    pub fn new(_bus: u8, address: u16) -> Self {
        Self {
            address,
            #[cfg(feature = "sensors-hw")]
            i2c: rppal::i2c::I2c::with_bus(_bus).expect("Failed to open I2C bus"),
            dig_t1: 0,
            dig_t2: 0,
            dig_t3: 0,
            dig_h1: 0,
            dig_h2: 0,
            dig_h3: 0,
            dig_h4: 0,
            dig_h5: 0,
            dig_h6: 0,
            dig_p1: 0,
            dig_p2: 0,
            dig_p3: 0,
            dig_p4: 0,
            dig_p5: 0,
            dig_p6: 0,
            dig_p7: 0,
            dig_p8: 0,
            dig_p9: 0,
            initialized: false,
        }
    }

    /// Read raw registers from BME280
    #[cfg(feature = "sensors-hw")]
    fn read_register(&mut self, reg: u8, buf: &mut [u8]) -> Result<(), SensorError> {
        self.i2c
            .write(&[reg])
            .map_err(|e| SensorError::I2c(format!("Write reg 0x{:02X}: {}", reg, e)))?;
        self.i2c
            .read(buf)
            .map_err(|e| SensorError::I2c(format!("Read reg 0x{:02X}: {}", reg, e)))?;
        Ok(())
    }

    #[cfg(feature = "sensors-hw")]
    fn write_register(&mut self, reg: u8, value: u8) -> Result<(), SensorError> {
        self.i2c
            .write(&[reg, value])
            .map_err(|e| SensorError::I2c(format!("Write 0x{:02X}=0x{:02X}: {}", reg, value, e)))?;
        Ok(())
    }

    #[cfg(feature = "sensors-hw")]
    fn read_calibration(&mut self) -> Result<(), SensorError> {
        let mut cal = [0u8; 26];
        self.read_register(0x88, &mut cal)?;

        self.dig_t1 = u16::from_le_bytes([cal[0], cal[1]]);
        self.dig_t2 = i16::from_le_bytes([cal[2], cal[3]]);
        self.dig_t3 = i16::from_le_bytes([cal[4], cal[5]]);
        self.dig_p1 = u16::from_le_bytes([cal[6], cal[7]]);
        self.dig_p2 = i16::from_le_bytes([cal[8], cal[9]]);
        self.dig_p3 = i16::from_le_bytes([cal[10], cal[11]]);
        self.dig_p4 = i16::from_le_bytes([cal[12], cal[13]]);
        self.dig_p5 = i16::from_le_bytes([cal[14], cal[15]]);
        self.dig_p6 = i16::from_le_bytes([cal[16], cal[17]]);
        self.dig_p7 = i16::from_le_bytes([cal[18], cal[19]]);
        self.dig_p8 = i16::from_le_bytes([cal[20], cal[21]]);
        self.dig_p9 = i16::from_le_bytes([cal[22], cal[23]]);

        let mut h1 = [0u8; 1];
        self.read_register(0xA1, &mut h1)?;
        self.dig_h1 = h1[0];

        let mut hcal = [0u8; 7];
        self.read_register(0xE1, &mut hcal)?;
        self.dig_h2 = i16::from_le_bytes([hcal[0], hcal[1]]);
        self.dig_h3 = hcal[2];
        self.dig_h4 = ((hcal[3] as i16) << 4) | ((hcal[4] & 0x0F) as i16);
        self.dig_h5 = ((hcal[5] as i16) << 4) | ((hcal[4] >> 4) as i16);
        self.dig_h6 = hcal[6] as i8;

        Ok(())
    }

    /// Compensate raw temperature to °C × 100 (integer)
    #[allow(dead_code)]
    fn compensate_temperature(&self, adc_t: i32) -> (i32, i32) {
        let var1 = (((adc_t >> 3) - ((self.dig_t1 as i32) << 1)) * (self.dig_t2 as i32)) >> 11;
        let var2 = (((((adc_t >> 4) - (self.dig_t1 as i32))
            * ((adc_t >> 4) - (self.dig_t1 as i32)))
            >> 12)
            * (self.dig_t3 as i32))
            >> 14;
        let t_fine = var1 + var2;
        let temperature = (t_fine * 5 + 128) >> 8; // °C × 100
        (temperature, t_fine)
    }

    /// Compensate raw humidity to % × 100 (integer)
    #[allow(dead_code)]
    #[inline(always)]
    fn compensate_humidity(&self, adc_h: i32, t_fine: i32) -> i32 {
        let mut var = t_fine - 76800_i32;
        if var == 0 {
            return 0;
        }
        var = ((((adc_h << 14) - ((self.dig_h4 as i32) << 20) - ((self.dig_h5 as i32) * var))
            + 16384)
            >> 15)
            * (((((((var * (self.dig_h6 as i32)) >> 10)
                * (((var * (self.dig_h3 as i32)) >> 11) + 32768))
                >> 10)
                + 2097152)
                * (self.dig_h2 as i32)
                + 8192)
                >> 14);
        var -= ((((var >> 15) * (var >> 15)) >> 7) * (self.dig_h1 as i32)) >> 4;
        if var < 0 {
            var = 0;
        }
        if var > 419430400 {
            var = 419430400;
        }
        (var >> 12) * 100 >> 10 // % × 100 (>>10 == /1024)
    }

    /// Compensate raw pressure to hPa × 10 (integer)
    #[allow(dead_code)]
    #[inline(always)]
    fn compensate_pressure(&self, adc_p: i32, t_fine: i32) -> i32 {
        let mut var1 = (t_fine as i64) - 128000;
        let mut var2 = var1 * var1 * (self.dig_p6 as i64);
        var2 += (var1 * (self.dig_p5 as i64)) << 17;
        var2 += (self.dig_p4 as i64) << 35;
        var1 = ((var1 * var1 * (self.dig_p3 as i64)) >> 8) + ((var1 * (self.dig_p2 as i64)) << 12);
        var1 = ((1_i64 << 47) + var1) * (self.dig_p1 as i64) >> 33;
        if var1 == 0 {
            return 0;
        }
        let mut p: i64 = 1048576 - adc_p as i64;
        p = (((p << 31) - var2) * 3125) / var1;
        var1 = ((self.dig_p9 as i64) * (p >> 13) * (p >> 13)) >> 25;
        var2 = ((self.dig_p8 as i64) * p) >> 19;
        p = ((p + var1 + var2) >> 8) + ((self.dig_p7 as i64) << 4);
        ((p >> 8) / 10) as i32 // hPa × 10 (>>8 == /256, /10 == *10/100 collapsed)
    }
}

impl SensorDriver for Bme280Sensor {
    fn init(&mut self) -> Result<(), SensorError> {
        #[cfg(feature = "sensors-hw")]
        {
            self.i2c.set_slave_address(self.address).map_err(|e| {
                SensorError::I2c(format!("Set address 0x{:02X}: {}", self.address, e))
            })?;

            // Check chip ID
            let mut id = [0u8; 1];
            self.read_register(0xD0, &mut id)?;
            if id[0] != 0x60 {
                return Err(SensorError::NotFound(format!(
                    "BME280 not found at 0x{:02X} (chip ID: 0x{:02X})",
                    self.address, id[0]
                )));
            }

            // Reset
            self.write_register(0xE0, 0xB6)?;
            std::thread::sleep(Duration::from_millis(10));

            // Read calibration
            self.read_calibration()?;

            // Configure: humidity oversampling ×1
            self.write_register(0xF2, 0x01)?;
            // Configure: temp ×1, pressure ×1, normal mode
            self.write_register(0xF4, 0x27)?;
            // Configure: standby 1000ms, filter off
            self.write_register(0xF5, 0xA0)?;

            std::thread::sleep(Duration::from_millis(50));
        }
        self.initialized = true;
        Ok(())
    }

    fn read_samples_interval(
        &mut self,
        count: usize,
        interval: Duration,
    ) -> Result<SensorBatch, SensorError> {
        if !self.initialized {
            self.init()?;
        }

        let mut batch = SensorBatch::new("BME280");
        let start = Instant::now();

        for _ in 0..count {
            let ts = start.elapsed().as_millis() as u64;

            #[cfg(feature = "sensors-hw")]
            {
                let mut raw = [0u8; 8];
                self.read_register(0xF7, &mut raw)?;

                let adc_p =
                    ((raw[0] as i32) << 12) | ((raw[1] as i32) << 4) | ((raw[2] as i32) >> 4);
                let adc_t =
                    ((raw[3] as i32) << 12) | ((raw[4] as i32) << 4) | ((raw[5] as i32) >> 4);
                let adc_h = ((raw[6] as i32) << 8) | (raw[7] as i32);

                let (temp, t_fine) = self.compensate_temperature(adc_t);
                let hum = self.compensate_humidity(adc_h, t_fine);
                let pres = self.compensate_pressure(adc_p, t_fine);

                batch.temperature.push(temp);
                batch.humidity.push(hum);
                batch.pressure.push(pres);
            }

            #[cfg(not(feature = "sensors-hw"))]
            {
                // Simulated data for testing without hardware
                const INV_1000: f32 = 1.0 / 1000.0;
                let t = ts as f32 * INV_1000;
                batch.temperature.push(2500 + (t * 10.0) as i32); // 25.00°C rising
                batch.humidity.push(6500 - (t * 5.0) as i32); // 65.00% falling
                batch.pressure.push(10132 + (t * 2.0) as i32); // 1013.2 hPa rising
            }

            batch.timestamps.push(ts);
            std::thread::sleep(interval);
        }

        Ok(batch)
    }

    fn name(&self) -> &'static str {
        "BME280"
    }
}

// ============================================================
// DHT22 — Temperature / Humidity (GPIO, single-wire)
// ============================================================

/// DHT22 (AM2302) GPIO sensor driver
///
/// Single-wire protocol, temperature and humidity.
///
/// # Pi 5 Wiring
///
/// | DHT22 | Pi 5 GPIO |
/// |-------|-----------|
/// | VCC | 3.3V (Pin 1) |
/// | DATA | GPIO 4 (Pin 7) + 10kΩ pull-up to VCC |
/// | GND | GND (Pin 6) |
#[allow(dead_code)]
pub struct Dht22Sensor {
    gpio_pin: u8,
    initialized: bool,
}

impl Dht22Sensor {
    pub fn new(gpio_pin: u8) -> Self {
        Self {
            gpio_pin,
            initialized: false,
        }
    }
}

impl SensorDriver for Dht22Sensor {
    fn init(&mut self) -> Result<(), SensorError> {
        #[cfg(feature = "sensors-hw")]
        {
            let gpio = rppal::gpio::Gpio::new()
                .map_err(|e| SensorError::Gpio(format!("GPIO init: {}", e)))?;
            let _pin = gpio
                .get(self.gpio_pin)
                .map_err(|e| SensorError::Gpio(format!("GPIO pin {}: {}", self.gpio_pin, e)))?;
        }
        self.initialized = true;
        Ok(())
    }

    fn read_samples_interval(
        &mut self,
        count: usize,
        interval: Duration,
    ) -> Result<SensorBatch, SensorError> {
        if !self.initialized {
            self.init()?;
        }

        let mut batch = SensorBatch::new("DHT22");
        let start = Instant::now();

        for _ in 0..count {
            let ts = start.elapsed().as_millis() as u64;

            #[cfg(feature = "sensors-hw")]
            {
                // DHT22 single-wire protocol:
                // 1. Pull data line low for 1ms
                // 2. Release and wait for sensor response
                // 3. Read 40 bits: 16 humidity + 16 temperature + 8 checksum
                let gpio =
                    rppal::gpio::Gpio::new().map_err(|e| SensorError::Gpio(format!("{}", e)))?;
                let mut pin = gpio
                    .get(self.gpio_pin)
                    .map_err(|e| SensorError::Gpio(format!("{}", e)))?
                    .into_io(rppal::gpio::Mode::Output);

                // Start signal
                pin.set_low();
                std::thread::sleep(Duration::from_millis(1));
                pin.set_mode(rppal::gpio::Mode::Input);

                // Read 40 bits (simplified - real implementation needs precise timing)
                let mut data = [0u8; 5];
                std::thread::sleep(Duration::from_micros(80)); // Wait for response

                for byte in 0..5 {
                    for bit in (0..8).rev() {
                        // Wait for high
                        let start_wait = Instant::now();
                        while pin.is_low() {
                            if start_wait.elapsed() > Duration::from_micros(100) {
                                return Err(SensorError::Timeout);
                            }
                        }
                        // Measure high duration
                        let high_start = Instant::now();
                        while pin.is_high() {
                            if high_start.elapsed() > Duration::from_micros(100) {
                                break;
                            }
                        }
                        if high_start.elapsed() > Duration::from_micros(40) {
                            data[byte] |= 1 << bit;
                        }
                    }
                }

                // Verify checksum
                let checksum =
                    (data[0] as u16 + data[1] as u16 + data[2] as u16 + data[3] as u16) & 0xFF;
                if checksum != data[4] as u16 {
                    return Err(SensorError::InvalidData("DHT22 checksum mismatch".into()));
                }

                let humidity = ((data[0] as i32) << 8 | data[1] as i32) * 10; // ×100
                let mut temp = ((data[2] as i32 & 0x7F) << 8 | data[3] as i32) * 10; // ×100
                if data[2] & 0x80 != 0 {
                    temp = -temp;
                }

                batch.temperature.push(temp);
                batch.humidity.push(humidity);
            }

            #[cfg(not(feature = "sensors-hw"))]
            {
                const INV_1000: f32 = 1.0 / 1000.0;
                let t = ts as f32 * INV_1000;
                batch.temperature.push(2200 + (t * 15.0) as i32);
                batch.humidity.push(7000 - (t * 8.0) as i32);
            }

            batch.timestamps.push(ts);

            // DHT22 minimum interval is 2 seconds
            let wait = if interval < Duration::from_secs(2) {
                Duration::from_secs(2)
            } else {
                interval
            };
            std::thread::sleep(wait);
        }

        Ok(batch)
    }

    fn name(&self) -> &'static str {
        "DHT22"
    }
}

// ============================================================
// ADXL345 — 3-Axis Accelerometer (SPI)
// ============================================================

/// ADXL345 SPI accelerometer driver
///
/// 3-axis digital accelerometer, ±2g to ±16g range.
///
/// # Pi 5 Wiring
///
/// | ADXL345 | Pi 5 GPIO |
/// |---------|-----------|
/// | VCC | 3.3V (Pin 1) |
/// | GND | GND (Pin 6) |
/// | CS | GPIO 8 / CE0 (Pin 24) |
/// | SDO | GPIO 9 / MISO (Pin 21) |
/// | SDA | GPIO 10 / MOSI (Pin 19) |
/// | SCL | GPIO 11 / SCLK (Pin 23) |
pub struct Adxl345Sensor {
    initialized: bool,
}

impl Adxl345Sensor {
    pub fn new() -> Self {
        Self { initialized: false }
    }
}

impl SensorDriver for Adxl345Sensor {
    fn init(&mut self) -> Result<(), SensorError> {
        #[cfg(feature = "sensors-hw")]
        {
            let mut spi = rppal::spi::Spi::new(
                rppal::spi::Bus::Spi0,
                rppal::spi::SlaveSelect::Ss0,
                5_000_000, // 5 MHz
                rppal::spi::Mode::Mode3,
            )
            .map_err(|e| SensorError::Spi(format!("SPI init: {}", e)))?;

            // Read device ID (should be 0xE5)
            let mut buf = [0x80 | 0x00, 0x00]; // Read reg 0x00
            spi.transfer(&mut buf)
                .map_err(|e| SensorError::Spi(format!("SPI transfer: {}", e)))?;
            if buf[1] != 0xE5 {
                return Err(SensorError::NotFound(format!(
                    "ADXL345 not found (ID: 0x{:02X})",
                    buf[1]
                )));
            }

            // Set measurement mode
            spi.write(&[0x2D, 0x08])
                .map_err(|e| SensorError::Spi(format!("Set measure mode: {}", e)))?;
            // Set ±2g range, full resolution
            spi.write(&[0x31, 0x08])
                .map_err(|e| SensorError::Spi(format!("Set range: {}", e)))?;
        }
        self.initialized = true;
        Ok(())
    }

    fn read_samples_interval(
        &mut self,
        count: usize,
        interval: Duration,
    ) -> Result<SensorBatch, SensorError> {
        if !self.initialized {
            self.init()?;
        }

        let mut batch = SensorBatch::new("ADXL345");
        let start = Instant::now();

        for _ in 0..count {
            let ts = start.elapsed().as_millis() as u64;

            #[cfg(feature = "sensors-hw")]
            {
                let mut spi = rppal::spi::Spi::new(
                    rppal::spi::Bus::Spi0,
                    rppal::spi::SlaveSelect::Ss0,
                    5_000_000,
                    rppal::spi::Mode::Mode3,
                )
                .map_err(|e| SensorError::Spi(format!("{}", e)))?;

                // Read 6 bytes from DATAX0 (0x32) with multi-byte flag
                let mut buf = [0u8; 7];
                buf[0] = 0x80 | 0x40 | 0x32; // Read + multi-byte + DATAX0
                spi.transfer(&mut buf)
                    .map_err(|e| SensorError::Spi(format!("{}", e)))?;

                let x = i16::from_le_bytes([buf[1], buf[2]]) as i32 * 4; // milli-g
                let y = i16::from_le_bytes([buf[3], buf[4]]) as i32 * 4;
                let z = i16::from_le_bytes([buf[5], buf[6]]) as i32 * 4;

                batch.accel_x.push(x);
                batch.accel_y.push(y);
                batch.accel_z.push(z);
            }

            #[cfg(not(feature = "sensors-hw"))]
            {
                const INV_1000: f32 = 1.0 / 1000.0;
                let t = ts as f32 * INV_1000;
                batch.accel_x.push((t.sin() * 100.0) as i32);
                batch.accel_y.push((t.cos() * 100.0) as i32);
                batch.accel_z.push(1000 + (t * 5.0) as i32); // ~1g + drift
            }

            batch.timestamps.push(ts);
            std::thread::sleep(interval);
        }

        Ok(batch)
    }

    fn name(&self) -> &'static str {
        "ADXL345"
    }
}

// ============================================================
// GPS NMEA — Position (UART)
// ============================================================

/// GPS UART sensor (NMEA 0183 protocol)
///
/// Reads NMEA sentences from a serial GPS module.
///
/// # Pi 5 Wiring
///
/// | GPS Module | Pi 5 GPIO |
/// |------------|-----------|
/// | VCC | 3.3V (Pin 1) |
/// | GND | GND (Pin 6) |
/// | TX | GPIO 15 / RXD (Pin 10) |
/// | RX | GPIO 14 / TXD (Pin 8) |
#[allow(dead_code)]
pub struct GpsSensor {
    port_path: String,
    baud_rate: u32,
    initialized: bool,
}

impl GpsSensor {
    pub fn new(port_path: &str, baud_rate: u32) -> Self {
        Self {
            port_path: port_path.to_string(),
            baud_rate,
            initialized: false,
        }
    }

    /// Parse NMEA GGA sentence for position data
    #[allow(dead_code)]
    fn parse_gga(sentence: &str) -> Option<(i32, i32, i32)> {
        let parts: Vec<&str> = sentence.split(',').collect();
        if parts.len() < 10 || !parts[0].ends_with("GGA") {
            return None;
        }

        // Latitude: ddmm.mmmm
        const INV_100: f64 = 1.0 / 100.0;
        const INV_60: f64 = 1.0 / 60.0;
        let lat_raw = parts[2].parse::<f64>().ok()?;
        let lat_deg = (lat_raw * INV_100).floor();
        let lat_min = lat_raw - lat_deg * 100.0;
        let mut lat = ((lat_deg + lat_min * INV_60) * 1_000_000.0) as i32;
        if parts[3] == "S" {
            lat = -lat;
        }

        // Longitude: dddmm.mmmm
        let lon_raw = parts[4].parse::<f64>().ok()?;
        let lon_deg = (lon_raw * INV_100).floor();
        let lon_min = lon_raw - lon_deg * 100.0;
        let mut lon = ((lon_deg + lon_min * INV_60) * 1_000_000.0) as i32;
        if parts[5] == "W" {
            lon = -lon;
        }

        // Altitude in meters
        let alt = (parts[9].parse::<f64>().unwrap_or(0.0) * 100.0) as i32;

        Some((lat, lon, alt))
    }
}

impl SensorDriver for GpsSensor {
    fn init(&mut self) -> Result<(), SensorError> {
        #[cfg(feature = "sensors-hw")]
        {
            let _port = serialport::new(&self.port_path, self.baud_rate)
                .timeout(Duration::from_secs(2))
                .open()
                .map_err(|e| SensorError::Uart(format!("Open {}: {}", self.port_path, e)))?;
        }
        self.initialized = true;
        Ok(())
    }

    fn read_samples_interval(
        &mut self,
        count: usize,
        interval: Duration,
    ) -> Result<SensorBatch, SensorError> {
        if !self.initialized {
            self.init()?;
        }

        let mut batch = SensorBatch::new("GPS");
        let start = Instant::now();

        #[cfg(feature = "sensors-hw")]
        {
            let mut port = serialport::new(&self.port_path, self.baud_rate)
                .timeout(Duration::from_secs(2))
                .open()
                .map_err(|e| SensorError::Uart(format!("{}", e)))?;

            let mut line_buf = String::new();
            let mut samples_read = 0;

            while samples_read < count {
                let mut byte = [0u8; 1];
                match port.read(&mut byte) {
                    Ok(1) => {
                        if byte[0] == b'\n' {
                            if let Some((lat, lon, alt)) = Self::parse_gga(&line_buf) {
                                let ts = start.elapsed().as_millis() as u64;
                                batch.latitude.push(lat);
                                batch.longitude.push(lon);
                                batch.altitude.push(alt);
                                batch.timestamps.push(ts);
                                samples_read += 1;
                            }
                            line_buf.clear();
                        } else if byte[0] != b'\r' {
                            line_buf.push(byte[0] as char);
                        }
                    }
                    _ => {}
                }
            }
        }

        #[cfg(not(feature = "sensors-hw"))]
        {
            // Simulated GPS data (Tokyo, moving north)
            for i in 0..count {
                let ts = start.elapsed().as_millis() as u64;
                batch.latitude.push(35_681_236 + i as i32 * 10); // ~35.681°N
                batch.longitude.push(139_767_125 + i as i32 * 5); // ~139.767°E
                batch.altitude.push(4000 + i as i32 * 2); // ~40m
                batch.timestamps.push(ts);
                std::thread::sleep(interval);
            }
        }

        Ok(batch)
    }

    fn name(&self) -> &'static str {
        "GPS"
    }
}

// ============================================================
// Simulated Sensor — For testing without hardware
// ============================================================

/// Simulated sensor for testing and demos
///
/// Generates realistic sensor data patterns without actual hardware.
/// Useful for development on macOS/x86 and CI/CD pipelines.
pub struct SimulatedSensor {
    /// Noise amplitude (raw integer units)
    pub noise_amplitude: i32,
    /// Base temperature (°C × 100)
    pub base_temp: i32,
    /// Temperature drift rate per sample (°C × 100 / 100)
    pub temp_drift: i32,
}

impl SimulatedSensor {
    /// Create a new simulated sensor
    ///
    /// * `base_temp` - Base temperature in °C × 100 (e.g., 2500 = 25.00°C)
    /// * `noise_amplitude` - Maximum noise amplitude (e.g., 5 = ±0.05°C)
    pub fn new(base_temp: i32, noise_amplitude: i32) -> Self {
        Self {
            noise_amplitude,
            base_temp,
            temp_drift: 10,
        }
    }

    /// Simple deterministic pseudo-random noise
    #[inline(always)]
    fn noise(seed: u64) -> i32 {
        let hash = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0x6A09E667);
        ((hash >> 48) as i32) % 50 - 25 // ±25
    }
}

impl SensorDriver for SimulatedSensor {
    fn init(&mut self) -> Result<(), SensorError> {
        Ok(())
    }

    fn read_samples_interval(
        &mut self,
        count: usize,
        interval: Duration,
    ) -> Result<SensorBatch, SensorError> {
        let mut batch = SensorBatch::new("Simulated");
        let start = Instant::now();

        const INV_100: f32 = 1.0 / 100.0;
        for i in 0..count {
            let ts = start.elapsed().as_millis() as u64;
            let t = i as f32;
            let noise = if self.noise_amplitude > 0 {
                Self::noise(i as u64) * self.noise_amplitude / 25
            } else {
                0
            };

            batch
                .temperature
                .push(self.base_temp + (t * self.temp_drift as f32 * INV_100) as i32 + noise);
            batch.humidity.push(6500 - (t * 5.0) as i32 + noise / 2);
            batch.pressure.push(10132 + (t * 2.0) as i32 + noise / 3);
            batch.accel_x.push(((t * 0.1).sin() * 100.0) as i32 + noise);
            batch.accel_y.push(((t * 0.1).cos() * 100.0) as i32 + noise);
            batch.accel_z.push(1000 + noise);
            batch.timestamps.push(ts);

            std::thread::sleep(interval);
        }

        Ok(batch)
    }

    fn name(&self) -> &'static str {
        "Simulated"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_sensor() {
        let mut sensor = SimulatedSensor::new(2500, 5);
        let batch = sensor.read_samples(10).unwrap();
        assert_eq!(batch.temperature.len(), 10);
        assert_eq!(batch.humidity.len(), 10);
        assert_eq!(batch.pressure.len(), 10);
        assert_eq!(batch.sensor_id, "Simulated");
    }

    #[test]
    fn test_bme280_compensate() {
        let mut sensor = Bme280Sensor::new(1, 0x76);
        sensor.dig_t1 = 27504;
        sensor.dig_t2 = 26435;
        sensor.dig_t3 = -1000;
        sensor.dig_h1 = 75;
        sensor.dig_h2 = 370;
        sensor.dig_h3 = 0;
        sensor.dig_h4 = 313;
        sensor.dig_h5 = 50;
        sensor.dig_h6 = 30;
        sensor.dig_p1 = 36477;
        sensor.dig_p2 = -10685;
        sensor.dig_p3 = 3024;
        sensor.dig_p4 = 2855;
        sensor.dig_p5 = 140;
        sensor.dig_p6 = -7;
        sensor.dig_p7 = 15500;
        sensor.dig_p8 = -14600;
        sensor.dig_p9 = 6000;
        let (temp, _t_fine) = sensor.compensate_temperature(519888);
        assert!(temp > 2000 && temp < 3500); // ~20-35°C range
    }

    #[test]
    fn test_gps_parse_gga() {
        let sentence = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,";
        let result = GpsSensor::parse_gga(sentence);
        assert!(result.is_some());
        let (lat, lon, alt) = result.unwrap();
        assert!(lat > 48_000_000); // ~48°N
        assert!(lon > 11_000_000); // ~11°E
        assert_eq!(alt, 54540); // 545.4m
    }
}
