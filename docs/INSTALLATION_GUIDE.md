# Installation & Setup Guide (macOS Apple Silicon M2)

**Institution:** Sapthagiri NPS University  
**Subject:** Big Data Computing Mini Project  
**Author / Candidate:** Vikas  
**Target Environment:** macOS (Apple Silicon M1/M2/M3/M4, 16 GB Unified RAM)  

---

## 1. Prerequisites Checklist

Before running the platform, ensure the following core tools are installed on your Mac:

1. **Homebrew** (macOS Package Manager):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. **Java Runtime Environment (Java 17 or Java 21 LTS)**:
   ```bash
   brew install openjdk@17
   # or
   brew install --cask temurin@17
   ```
3. **Apache Hadoop 3.5.0**:
   ```bash
   brew install hadoop
   ```
4. **Python 3.12 or 3.14**:
   ```bash
   brew install python@3.12
   ```

---

## 2. Project Virtual Environment Setup

From the repository root (`/Users/vikas/Desktop/BDT`):

```bash
# 1. Create Python virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Install dashboard requirements
pip install -r requirements-dashboard.txt

# 6. Install the project in editable mode
pip install -e .
```

---

## 3. Hadoop HDFS & YARN Configuration

Hadoop on Apple Silicon installs to `/opt/homebrew/opt/hadoop/libexec`. Verify configuration in `/opt/homebrew/opt/hadoop/libexec/etc/hadoop`:

### 3.1 `core-site.xml`
```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/Users/vikas/hadoop_data/tmp</value>
    </property>
</configuration>
```

### 3.2 `hdfs-site.xml`
```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:///Users/vikas/hadoop_data/hdfs/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:///Users/vikas/hadoop_data/hdfs/datanode</value>
    </property>
</configuration>
```

### 3.3 NameNode Initial Formatting (First Time Only)
```bash
# Only run once if setting up from scratch:
hdfs namenode -format
```

---

## 4. Environment Variables Configuration

The project includes `scripts/env.sh` which automatically configures paths for Hadoop, Spark, Java, and Python.

To test the environment configuration:
```bash
source scripts/env.sh
echo "JAVA_HOME: $JAVA_HOME"
echo "HADOOP_HOME: $HADOOP_HOME"
echo "PYSPARK_PYTHON: $PYSPARK_PYTHON"
```

---

## 5. Verification of Installation

Run the automated status check script:
```bash
bash scripts/hadoop_status.sh
```

Expected output:
- Active Java processes: `NameNode`, `DataNode`, `ResourceManager`, `NodeManager`
- Live HDFS capacity: ~228 GB
- Active YARN node: 1 Running Node
- PySpark version: `3.5.9`

You are now ready to run the project.
