//! OTA (Over-The-Air) アップデート
//!
//! ファームウェアのリモート更新管理。
//! バージョンチェック、ダウンロード検証、適用。

/// OTA 設定。
#[derive(Debug, Clone)]
pub struct OtaConfig {
    /// アップデートエンドポイント URL。
    pub endpoint: String,
    /// チェック間隔 (秒)。
    pub check_interval_secs: u64,
    /// 署名検証を有効にするか。
    pub verify_signature: bool,
    /// 最大ファームウェアサイズ (バイト)。
    pub max_firmware_size: u64,
    /// 自動適用。
    pub auto_apply: bool,
    /// デバイス ID。
    pub device_id: String,
    /// 現在のバージョン。
    pub current_version: String,
}

impl Default for OtaConfig {
    fn default() -> Self {
        Self {
            endpoint: String::from("https://ota.example.com/api/v1"),
            check_interval_secs: 3600,
            verify_signature: true,
            max_firmware_size: 16 * 1024 * 1024, // 16 MB
            auto_apply: false,
            device_id: String::new(),
            current_version: String::from("0.0.0"),
        }
    }
}

/// ファームウェア情報。
#[derive(Debug, Clone)]
pub struct FirmwareInfo {
    /// バージョン。
    pub version: String,
    /// ファイルサイズ (バイト)。
    pub size: u64,
    /// SHA-256 ハッシュ (hex)。
    pub sha256: String,
    /// 署名データ (オプション)。
    pub signature: Option<Vec<u8>>,
    /// ダウンロード URL。
    pub download_url: String,
    /// リリースノート。
    pub release_notes: String,
}

impl FirmwareInfo {
    /// バージョンが現在より新しいか判定。
    #[must_use]
    pub fn is_newer_than(&self, current: &str) -> bool {
        compare_versions(&self.version, current) > 0
    }
}

/// OTA 状態。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtaState {
    /// 待機中。
    Idle,
    /// 更新チェック中。
    Checking,
    /// ダウンロード中。
    Downloading,
    /// 検証中。
    Verifying,
    /// インストール中。
    Installing,
    /// エラー。
    Error,
    /// 完了。
    Complete,
}

impl core::fmt::Display for OtaState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Checking => write!(f, "Checking"),
            Self::Downloading => write!(f, "Downloading"),
            Self::Verifying => write!(f, "Verifying"),
            Self::Installing => write!(f, "Installing"),
            Self::Error => write!(f, "Error"),
            Self::Complete => write!(f, "Complete"),
        }
    }
}

/// OTA マネージャ。
#[derive(Debug)]
pub struct OtaManager {
    /// 設定。
    config: OtaConfig,
    /// 現在の状態。
    state: OtaState,
    /// ダウンロード済みファームウェアデータ。
    firmware_data: Option<Vec<u8>>,
    /// 利用可能なアップデート情報。
    available_update: Option<FirmwareInfo>,
    /// ダウンロード進捗 (0-100)。
    progress: u8,
    /// 更新履歴。
    history: Vec<UpdateRecord>,
}

/// アップデート履歴レコード。
#[derive(Debug, Clone)]
pub struct UpdateRecord {
    /// バージョン。
    pub version: String,
    /// タイムスタンプ (Unix 秒)。
    pub timestamp: u64,
    /// 成功したか。
    pub success: bool,
}

impl OtaManager {
    /// 新しい OTA マネージャを作成。
    #[must_use]
    pub const fn new(config: OtaConfig) -> Self {
        Self {
            config,
            state: OtaState::Idle,
            firmware_data: None,
            available_update: None,
            progress: 0,
            history: Vec::new(),
        }
    }

    /// 現在の状態。
    #[must_use]
    pub const fn state(&self) -> OtaState {
        self.state
    }

    /// ダウンロード進捗。
    #[must_use]
    pub const fn progress(&self) -> u8 {
        self.progress
    }

    /// 利用可能なアップデート情報。
    #[must_use]
    pub const fn available_update(&self) -> Option<&FirmwareInfo> {
        self.available_update.as_ref()
    }

    /// アップデートチェックを開始 (結果をセット)。
    pub fn check_update(&mut self, info: Option<FirmwareInfo>) {
        self.state = OtaState::Checking;

        if let Some(fw) = info {
            if fw.is_newer_than(&self.config.current_version) {
                if fw.size <= self.config.max_firmware_size {
                    self.available_update = Some(fw);
                    self.state = OtaState::Idle;
                } else {
                    self.state = OtaState::Error;
                }
            } else {
                self.available_update = None;
                self.state = OtaState::Idle;
            }
        } else {
            self.available_update = None;
            self.state = OtaState::Idle;
        }
    }

    /// ファームウェアデータを受信 (シミュレーション)。
    pub fn receive_firmware(&mut self, data: Vec<u8>) {
        self.state = OtaState::Downloading;
        self.progress = 100;
        self.firmware_data = Some(data);
        self.state = OtaState::Verifying;
    }

    /// ファームウェアを検証。
    ///
    /// # Errors
    ///
    /// ファームウェアがない、サイズ超過、またはハッシュ不一致の場合。
    pub fn verify_firmware(&mut self) -> Result<(), OtaError> {
        let data = self.firmware_data.as_ref().ok_or(OtaError::NoFirmware)?;
        let info = self
            .available_update
            .as_ref()
            .ok_or(OtaError::NoUpdateAvailable)?;

        // サイズチェック
        if data.len() as u64 != info.size {
            self.state = OtaState::Error;
            return Err(OtaError::SizeMismatch {
                expected: info.size,
                actual: data.len() as u64,
            });
        }

        // SHA-256 検証 (簡易: データの長さベースハッシュ)
        let computed_hash = simple_sha256_hex(data);
        if computed_hash != info.sha256 {
            self.state = OtaState::Error;
            return Err(OtaError::HashMismatch);
        }

        self.state = OtaState::Idle;
        Ok(())
    }

    /// アップデートを適用。
    ///
    /// # Errors
    ///
    /// ファームウェアがない場合。
    pub fn apply_update(&mut self, now_unix: u64) -> Result<(), OtaError> {
        if self.firmware_data.is_none() {
            return Err(OtaError::NoFirmware);
        }

        self.state = OtaState::Installing;

        let version = self
            .available_update
            .as_ref()
            .map_or_else(|| "unknown".to_string(), |fw| fw.version.clone());

        self.history.push(UpdateRecord {
            version: version.clone(),
            timestamp: now_unix,
            success: true,
        });

        self.config.current_version = version;
        self.firmware_data = None;
        self.available_update = None;
        self.progress = 0;
        self.state = OtaState::Complete;

        Ok(())
    }

    /// 現在のバージョン。
    #[must_use]
    pub fn current_version(&self) -> &str {
        &self.config.current_version
    }

    /// 設定への参照。
    #[must_use]
    pub const fn config(&self) -> &OtaConfig {
        &self.config
    }

    /// 更新履歴。
    #[must_use]
    pub fn history(&self) -> &[UpdateRecord] {
        &self.history
    }
}

/// 簡易 SHA-256 ハッシュ (テスト用、非暗号安全)。
fn simple_sha256_hex(data: &[u8]) -> String {
    // FNV-1a 64bit をベースにした簡易ハッシュ (本番では sha2 クレートを使用)
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01B3);
    }
    format!("{hash:016x}")
}

/// バージョン文字列を比較 (semver-like)。
///
/// 戻り値: >0 なら a が新しい、<0 なら b が新しい、0 なら同じ。
fn compare_versions(a: &str, b: &str) -> i32 {
    let parse = |s: &str| -> Vec<u32> {
        s.split('.')
            .map(|p| p.parse::<u32>().unwrap_or(0))
            .collect()
    };

    let va = parse(a);
    let vb = parse(b);
    let max_len = va.len().max(vb.len());

    for i in 0..max_len {
        let pa = va.get(i).copied().unwrap_or(0);
        let pb = vb.get(i).copied().unwrap_or(0);
        if pa != pb {
            return if pa > pb { 1 } else { -1 };
        }
    }
    0
}

/// OTA エラー。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OtaError {
    /// ファームウェアデータがない。
    NoFirmware,
    /// 利用可能なアップデートがない。
    NoUpdateAvailable,
    /// サイズ不一致。
    SizeMismatch {
        /// 期待サイズ。
        expected: u64,
        /// 実際サイズ。
        actual: u64,
    },
    /// ハッシュ不一致。
    HashMismatch,
    /// 署名検証失敗。
    SignatureInvalid,
    /// ネットワークエラー。
    NetworkError(String),
}

impl core::fmt::Display for OtaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoFirmware => write!(f, "No firmware data"),
            Self::NoUpdateAvailable => write!(f, "No update available"),
            Self::SizeMismatch { expected, actual } => {
                write!(f, "Size mismatch: expected {expected}, got {actual}")
            }
            Self::HashMismatch => write!(f, "Hash mismatch"),
            Self::SignatureInvalid => write!(f, "Invalid signature"),
            Self::NetworkError(msg) => write!(f, "Network error: {msg}"),
        }
    }
}

impl std::error::Error for OtaError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = OtaConfig::default();
        assert_eq!(config.check_interval_secs, 3600);
        assert!(config.verify_signature);
        assert!(!config.auto_apply);
    }

    #[test]
    fn ota_state_display() {
        assert_eq!(OtaState::Idle.to_string(), "Idle");
        assert_eq!(OtaState::Downloading.to_string(), "Downloading");
        assert_eq!(OtaState::Complete.to_string(), "Complete");
    }

    #[test]
    fn version_comparison() {
        assert!(compare_versions("1.1.0", "1.0.0") > 0);
        assert!(compare_versions("1.0.0", "1.1.0") < 0);
        assert_eq!(compare_versions("1.0.0", "1.0.0"), 0);
        assert!(compare_versions("2.0.0", "1.9.9") > 0);
    }

    #[test]
    fn firmware_is_newer() {
        let fw = FirmwareInfo {
            version: "1.1.0".into(),
            size: 1000,
            sha256: String::new(),
            signature: None,
            download_url: String::new(),
            release_notes: String::new(),
        };
        assert!(fw.is_newer_than("1.0.0"));
        assert!(!fw.is_newer_than("2.0.0"));
    }

    #[test]
    fn ota_manager_new() {
        let mgr = OtaManager::new(OtaConfig::default());
        assert_eq!(mgr.state(), OtaState::Idle);
        assert_eq!(mgr.progress(), 0);
        assert!(mgr.available_update().is_none());
    }

    #[test]
    fn check_update_newer() {
        let mut mgr = OtaManager::new(OtaConfig {
            current_version: "1.0.0".into(),
            ..OtaConfig::default()
        });

        let fw = FirmwareInfo {
            version: "1.1.0".into(),
            size: 100,
            sha256: String::new(),
            signature: None,
            download_url: String::new(),
            release_notes: String::new(),
        };

        mgr.check_update(Some(fw));
        assert!(mgr.available_update().is_some());
    }

    #[test]
    fn check_update_same_version() {
        let mut mgr = OtaManager::new(OtaConfig {
            current_version: "1.0.0".into(),
            ..OtaConfig::default()
        });

        let fw = FirmwareInfo {
            version: "1.0.0".into(),
            size: 100,
            sha256: String::new(),
            signature: None,
            download_url: String::new(),
            release_notes: String::new(),
        };

        mgr.check_update(Some(fw));
        assert!(mgr.available_update().is_none());
    }

    #[test]
    fn check_update_too_large() {
        let mut mgr = OtaManager::new(OtaConfig {
            current_version: "1.0.0".into(),
            max_firmware_size: 50,
            ..OtaConfig::default()
        });

        let fw = FirmwareInfo {
            version: "2.0.0".into(),
            size: 100,
            sha256: String::new(),
            signature: None,
            download_url: String::new(),
            release_notes: String::new(),
        };

        mgr.check_update(Some(fw));
        assert_eq!(mgr.state(), OtaState::Error);
    }

    #[test]
    fn verify_no_firmware() {
        let mut mgr = OtaManager::new(OtaConfig::default());
        assert_eq!(mgr.verify_firmware(), Err(OtaError::NoFirmware));
    }

    #[test]
    fn verify_size_mismatch() {
        let mut mgr = OtaManager::new(OtaConfig {
            current_version: "1.0.0".into(),
            ..OtaConfig::default()
        });

        let fw = FirmwareInfo {
            version: "2.0.0".into(),
            size: 999,
            sha256: String::new(),
            signature: None,
            download_url: String::new(),
            release_notes: String::new(),
        };

        mgr.check_update(Some(fw));
        mgr.receive_firmware(vec![0u8; 10]);
        assert!(mgr.verify_firmware().is_err());
    }

    #[test]
    fn full_update_flow() {
        let data = vec![0xAB; 64];
        let hash = simple_sha256_hex(&data);

        let mut mgr = OtaManager::new(OtaConfig {
            current_version: "1.0.0".into(),
            ..OtaConfig::default()
        });

        let fw = FirmwareInfo {
            version: "2.0.0".into(),
            size: 64,
            sha256: hash,
            signature: None,
            download_url: String::new(),
            release_notes: String::new(),
        };

        mgr.check_update(Some(fw));
        mgr.receive_firmware(data);
        mgr.verify_firmware().unwrap();
        mgr.apply_update(1000).unwrap();

        assert_eq!(mgr.state(), OtaState::Complete);
        assert_eq!(mgr.current_version(), "2.0.0");
        assert_eq!(mgr.history().len(), 1);
    }

    #[test]
    fn apply_no_firmware() {
        let mut mgr = OtaManager::new(OtaConfig::default());
        assert_eq!(mgr.apply_update(0), Err(OtaError::NoFirmware));
    }

    #[test]
    fn ota_error_display() {
        assert_eq!(OtaError::NoFirmware.to_string(), "No firmware data");
        assert_eq!(OtaError::HashMismatch.to_string(), "Hash mismatch");
    }
}
