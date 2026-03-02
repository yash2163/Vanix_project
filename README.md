# **README: Bovine Heat Detection Intelligence (V1.0 Demo)**

**⚠️ DISCLAIMER:** This code is a **sample demonstration** performed on a specific subset of cattle data. It is intended to illustrate the core logic and pattern recognition principles. It does not represent the full production system. The actual implementation requires integration with the live backend data flow and must handle complex edge-case scenarios not covered in this demo.

**1. System Architecture & Flow**

The following logic follows the structure detailed in the **attached flow diagram**. The system is decoupled into five modular blocks to ensure scalability and ease of integration with a live server environment.

### **Block 1: Feature Extraction**

* **Method:** `extract_features(self, df)`  
* **Logic:** Uses **VeDBA** (Vectorial Dynamic Body Acceleration) to calculate the energy of movement by removing the static force of gravity.  
* **Purpose:** Ensures the system analyzes actual animal motion rather than sensor orientation or tilt.

### **Block 2: Activity Recognition (AR Model)**

* **Current State:** Uses a basic heuristic threshold (`self.RES_THRESHOLD = 0.35`) in `predict_activity(self, df)` (Lines 24-28).  
* **Future Integration:** Currently, restlessness is computed at a basic level. In the production version, this block will be more improved and also there will be a **trained ML AR Model**. This will allow for significantly higher accuracy by recognizing complex behavioral patterns (Mounting, Chin-resting, etc.) rather than just high-energy movement.

### **Block 3: Environmental Heat Stress & Signal Integrity**

* **Status:** Currently a standalone block (`check_heat_stress`).  
* **Logic:** Analyzes `SOLAR_HOURS` (11 AM – 4 PM) for extreme temperature spikes.  
* **Production Note:** While shown as a separate sample here, Block 3 and general Signal Integrity checks (sensor health) will be **submerged into the main core logic** to provide a unified "Safety Valve" that suppresses false alerts caused by weather or sensor noise.

### **Block 4: Cattle Logic (The Core Engine)**

* **Current Scope:** This version only includes the **Heat Detector** (`cattle_logic_engine`).  
* **Operational Logic:**  
  1. **Global Anchor:** Establishes a "True Zero" using the minimum night temperature across the entire dataset.  
  2. **Sequential Synchronization:** Uses a weighted score to find the specific sequence of a Night Spike followed by Daytime Restlessness.

   The main production code will include a **Disease Detector** (Fever/Mastitis) and **Anomaly Detection** using Circadian pattern analysis to identify deviations from a cow's historical normal.

### **Block 5: Alert & Logging Engine**

* **Function:** `generate_logs(self, results, peak_date)`.  
* **Ranking:** Uses a "Winner-Take-All" approach to ensure only the most clinically significant day is flagged as a "Confirmed Heat."

**2. Critical Analysis of Script Logs**

The logic was tested across two different temporal windows (January and February) for the same node (Cow 184). As per ground report, 13th was the actual heat cycle date.

### **Case Study: February 13th (True Heat)**

* **Pattern:** The log correctly identifies Feb 13th as the peak.  
* **Data Analysis:** Even though activity on Feb 11th was "louder" in terms of raw movement, Feb 13th won because it possessed the **highest Night Spike (9.11°C)** relative to the Global Anchor.  
* The system successfully prioritized the **Biological Trigger** (Temperature) over the **Behavioral Noise** (Movement).

### **Case Study: January 21st (Behavioral Dominance)**

* **Pattern:** Heat detected on Jan 21st.  
* **Data Analysis:** This log showed an extreme **Persistence of 83%**.  
* **Verdict:** In this specific case, the behavioral restlessness was so overwhelming and sustained that the system triggered a "Confirmed" alert despite a subtle temperature change.
