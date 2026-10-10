-- init_db.sql
-- 直接用 sqlite3 CLI 建表 + 插演示数据，无需 Python
-- 用途：测试 / 离线初始化 / 验收

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS screenings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    patient_name TEXT,
    age INTEGER,
    gender TEXT,
    diabetes INTEGER DEFAULT 0,
    screening_date TEXT NOT NULL,
    diagnosis TEXT,
    dr_grade INTEGER DEFAULT 0,
    cv_risk TEXT,
    htn_risk TEXT,
    ckd_risk TEXT,
    confidence REAL DEFAULT 0.0,
    gradcam_path TEXT,
    glaucoma_risk TEXT DEFAULT '低风险',
    glaucoma_prob REAL DEFAULT 0.0,
    amd_risk TEXT DEFAULT '无',
    amd_prob REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_screenings_pid ON screenings(patient_id);

-- 演示数据：5 个患者 / 10 条记录
INSERT OR IGNORE INTO screenings (id, patient_id, patient_name, age, gender, diabetes, screening_date, diagnosis, dr_grade, cv_risk, htn_risk, ckd_risk, confidence, glaucoma_risk, glaucoma_prob, amd_risk, amd_prob) VALUES
(1, 'P001', '张三', 62, '男', 1, '2025-01-01', 'DR 0 级',  0, '低', '低', '低', 0.92, '低风险', 0.10, '无',  0.05),
(2, 'P001', '张三', 62, '男', 1, '2025-04-01', 'DR 1 级',  1, '低', '低', '低', 0.85, '低风险', 0.12, '无',  0.08),
(3, 'P001', '张三', 62, '男', 1, '2025-07-01', 'DR 2 级',  2, '中', '中', '中', 0.88, '中风险', 0.55, '早期',0.45),
(4, 'P001', '张三', 62, '男', 1, '2025-10-01', 'DR 2 级',  2, '中', '中', '中', 0.90, '中风险', 0.60, '早期',0.50),
(5, 'P002', '李四', 55, '女', 0, '2025-01-31', 'DR 1 级',  1, '低', '中', '不适用（非糖尿病患者）', 0.82, '低风险', 0.18, '无', 0.10),
(6, 'P002', '李四', 55, '女', 0, '2025-06-30', 'DR 0 级',  0, '低', '低', '不适用（非糖尿病患者）', 0.91, '低风险', 0.10, '无', 0.05),
(7, 'P003', '王五', 71, '男', 1, '2025-03-02', 'DR 3 级',  3, '高', '高', '高', 0.80, '高风险', 0.85, '中期',0.70),
(8, 'P003', '王五', 71, '男', 1, '2025-08-20', 'DR 4 级',  4, '高', '高', '高', 0.78, '高风险', 0.92, '晚期',0.88),
(9, 'P004', '赵六', 48, '女', 0, '2025-02-15', 'DR 0 级',  0, '低', '低', '不适用（非糖尿病患者）', 0.94, '低风险', 0.08, '无', 0.04),
(10,'P005', '钱七', 67, '男', 1, '2025-03-17', 'DR 2 级',  2, '中', '中', '中', 0.86, '中风险', 0.50, '早期',0.42);
