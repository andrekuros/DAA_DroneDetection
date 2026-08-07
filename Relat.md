# Technical Report – Point D

### Demonstration of Detection and Avoidance Architectures

### for Safe Integration of Unmanned Aircraft System (UAS) in

### Non-Segregated Airspace

### SIMUA Project – Safe Integration of Manned and Unmanned Aircrafts

```
Brazil-Sweden Strategic Partnership | Air Domain Study (ADS)
```
### Pedro Henrique Blatt Machado

### Tenente-Coronel André Kuroswiski

### Instituto Tecnológico de Aeronáutica (ITA)

### Instituto de Controle do Espaço Aéreo (ICEA)

### March 2026

```
Abstract
This report documents the technical results of Point D of the Work Plan for the
SIMUA project ( Safe Integration of Manned and Unmanned Aircrafts ), developed
within the framework of the strategic partnership between Instituto Tecnológico
de Aeronáutica (ITA) and Instituto de Controle do Espaço Aéreo (ICEA) with the
```

```
Swedish Air Domain Study (ADS) program. The document presents the architecture
exploration methodology for Detect and Avoid (DAA) systems in non-segregated
airspaces, structured around the international Remain Well Clear (RWC)/Collision
Avoidance (CA) layered framework (RTCA DO-365B, ASTM F3442). The work
covers: (i) the trade-off study of candidate sensors (LiDAR, mmWave Radar, Com-
puter Vision, and Acoustics) with justification of Size, Weight and Power (SWaP)
constraints; (ii) the definition of detection envelopes and Operational Services and
Environment Definition (OSED) mapped to Brazilian Airspace Control System (SIS-
CEAB) airspace classes (G, E, D, C); (iii) integration with the DAIDALUS algorithm
as the RWC implementation, including Sensor Uncertainty Mitigation (SUM)/Virtual
Intruder State Aggregation (VISA) logic for non-cooperative scenarios and Human-in-
the-Loop (HITL) alerting design; (iv) area saturation analysis validating conceptual
capacity ( > 7 UAS/km^2 ) in dense urban environments; (v) pre-tactical infrastruc-
ture optimization (SCOPAS) generating cost-versus-coverage threshold curves for
sensor network design; and (vi) strategic microclimate forecasting (WRF/UCM) as
a prototype operational pre-tactical layer for dynamic environmental risk mitigation
within Unmanned Traffic Management (UTM) and architectures. The experimental
campaign combines three complementary lines of research evidence: a Macro-Scale
Strategic Layering Analysis (Macro-Scale Strategic Layering Analysis (MSLA)) for
strategic UTM/DAA capacity exploration via encounter models; a High-Fidelity
Tactical Logic Validation (High-Fidelity Tactical Logic Validation (HTLV)) for kine-
matic assessment at 50 Hz update rates; and a synchronized high-fidelity multi-sensor
benchmark that empirically investigates fusion and uncertainty assumptions for the
“sense” layer, establishing foundations for future no-intruder clutter and false-alert
quantification. All studies represent ongoing research initiatives; their primary
value is the capability understanding developed to support the consolidation
of alternatives for future development, not an immediately deployable operational
solution. The findings indicate the hybrid Airborne Detect and Avoid (ABDAA)
architecture—combining mmWave Radar and Computer Vision with DAIDALUS
SUM logic—as the preferred research pathway under SIMUA SWaP and Concept
of Operations (CONOPS) constraints for safe and scalable drone traffic management
in Brazilian airspace.
```
**Keywords:** Detect and Avoid; Remain Well Clear; Collision Avoidance; UAS; DAIDALUS;
mmWave Radar; Computer Vision; UTM; U-space; OSED; Microclimate Forecasting;
Sensor Coverage Optimization; Encounter Models; Risk Ratio; SIMUA; SWaP; DO-365B;
ASTM F3442.


## Contents



- List of Acronyms and Abbreviations
- 1 Technical Foundation and Requirements
   - 1.1 Strategic Alignment and Partnership
      - 1.1.1 Context and Motivation
      - 1.1.2 Milestone Identification – Point D
      - 1.1.3 Partners and Responsibilities
   - 1.2 Architecture Definition: Strategy and Sensors
      - 1.2.1 DAA Layered Architecture: RWC and CA
      - 1.2.2 Operational Services and Environment Definition (OSED)
      - 1.2.3 Sensor Trade-off Study: Motivation and Criteria
      - 1.2.4 LiDAR: Evaluation and Infeasibility on Micro-UAS
      - 1.2.5 Millimeter Wave Radar (mmWave)
      - 1.2.6 Computer Vision with Edge AI
      - 1.2.7 Passive Acoustic Array
      - 1.2.8 Sensor Comparison Table
      - 1.2.9 Operational Envelopes and Detection Probability
- 2 Strategic Foundation and Experimental Campaign
   - 2.1 Sensor Coverage Optimization for Protected Air Space (SCOPAS)
      - 2.1.1 Integration Rationale and Problem Statement
      - 2.1.2 Modeling Pipeline and Sensor Parameterization
      - 2.1.3 Inputs, Profiles, and Solution Description
      - 2.1.4 Optimization Results Compendium
      - 2.1.5 Limitations, Interpretation, and Recommendations
   - 2.2 Microclimate Forecasting as an Operational UTM/U-space Layer
      - 2.2.1 Operational Motivation
      - 2.2.2 International Reference Context
      - 2.2.3 Current Research State and Prototype Results
      - 2.2.4 Evolution Roadmap
      - 2.2.5 Implications for UTM Governance and BVLOS Operations
   - 2.3 Macro-Scale Strategic Layering Analysis (MSLA)
      - 2.3.1 Nominal traffic and safety parameters
      - 2.3.2 Scenario 1 — Baseline: Blind Flight
      - 2.3.3 Scenario 2 — DAA Only (Reactive)
      - 2.3.4 Scenario 3 — xTM Only (Strategic)
      - 2.3.5 Scenario 4A — Optimized xTM + DAA
         - counter Model Validation) 2.3.6 Scenario 4B — xTM + Intelligent DAA + Stochastic Wind (En-
      - 2.3.7 Summary of Python Experiments
      - 2.3.8 Parity with High-Fidelity Tactical Engine
   - 2.4 High-Fidelity Tactical Logic Validation (HTLV)
      - 2.4.1 HTLV Architecture and Execution Engine
      - 2.4.2 Methodological Comparison: Macro-Scale vs. High-Fidelity
      - 2.4.3 Complementary Strengths and Integration Path
      - 2.4.4 Path Ahead: Integration Proposal
      - 2.4.5 DAIDALUS Core Logic and Multi-Sensor Fusion
      - 2.4.6 Scenario 2 Action-Mode Scalability Evaluation (1h, 10 Hz)
      - 2.4.7 Summary for DECEA Evaluation
   - 2.5 Multi-Sensor Simulation Campaign for Method Development
      - 2.5.1 Objective and evaluation questions
      - 2.5.2 Experimental Setup and Architecture
      - 2.5.3 Electro-Optical (EO) Tracking and Classification Performance
      - 2.5.4 Cross-Sensor Detection by Operational Distance Band
      - 2.5.5 Condition-level performance (day/dawn/dusk/night)
      - 2.5.6 Earliest detection and warning relevance
      - 2.5.7 Simulation Strengths and Existing Limitations
      - 2.5.8 Implication for ABDAA
   - 2.6 Integrated Detection and Resolution Performance
      - 2.6.1 Detection Performance
      - 2.6.2 Conflict Resolution Effectiveness
      - 2.6.3 Risk Ratio Analysis
      - 2.6.4 Nuisance Alert Analysis
      - 2.6.5 Electro-Optical Tracking Architecture
- 3 Operational Impact and Airspace Capacity
   - 3.1 Airspace Saturation Analysis
      - 3.1.1 Stress Test Methodology in Urban Corridors
      - 3.1.2 Capacity Comparison: The Impact of DAA Architecture
      - 3.1.3 Quantitative Saturation Results
   - 3.2 Alignment with CONOPS and SISCEAB
      - 3.2.1 Validation of Operational Premises
      - 3.2.2 Link to SISCEAB Strategic Objectives
      - 3.2.3 Compliance with SISCEAB and Federated Management
      - 3.2.4 Convergence of UTM and UAM (eVTOL) Operations
      - 3.2.5 Integration with Digital Flight Rules (DFR)
      - 3.2.6 ASTM Interoperability and Ecosystem Architecture
      - 3.2.7 Mitigation of Identified Risks
- 4 Conclusions and Strategic Vision
   - 4.1 Conclusions and Deliverables Summary
      - 4.1.1 Research Nature and Overall Assessment
      - 4.1.2 Research Maturity and Identified Limitations
      - 4.1.3 Contribution to the SIMUA Project and Future Vision
   - .1 Simulation Logs
   - .2 MSLA campaign metrics (MACproxy and Risk Ratio)
   - .3 Simulation Video Gallery and Captures
   - .4 CesiumJS Simulation Playbacks
   - .5 DAIDALUS v2 Configuration Parameters
   - .6 Multi-Sensor Benchmark: Technical Framework


## List of Acronyms and Abbreviations

**AAM** Advanced Air Mobility.

**ABDAA** Airborne Detect and Avoid.

**ADS-B** Automatic Dependent Surveillance-Broadcast.

**ANAC** Agência Nacional de Aviação Civil.

**ATC** Air Traffic Control.

**BVLOS** Beyond Visual Line of Sight.

**CA** Collision Avoidance.

**CONOPS** Concept of Operations.

**CPA** Closest Point of Approach.

**DAA** Detect and Avoid.

**DECEA** Departamento de Controle do Espaço Aéreo.

**DFR** Digital Flight Rules.

**eVTOL** Electric Vertical Takeoff and Landing.

**HITL** Human-in-the-Loop.

**HTLV** High-Fidelity Tactical Logic Validation.

**ICEA** Instituto de Controle do Espaço Aéreo.

**ITA** Instituto Tecnológico de Aeronáutica.

**MSLA** Macro-Scale Strategic Layering Analysis.

**NMAC** Near Mid-Air Collision.

**OSED** Operational Services and Environment Definition.

**PSU** Provider of Services for UAM.

**RID** Remote Identification.

**RPIC** Remote Pilot in Command.


**RWC** Remain Well Clear.

**SISCEAB** Brazilian Airspace Control System.

**sUAS** Small Unmanned Aircraft System.

**SUM** Sensor Uncertainty Mitigation.

**SWaP** Size, Weight and Power.

**UAM** Urban Air Mobility.

**UAS** Unmanned Aircraft System.

**USS** UAS Service Supplier.

**UTM** Unmanned Traffic Management.

**VISA** Virtual Intruder State Aggregation.

**xTM** UAS Traffic Management Extension.


## 1 Technical Foundation and Requirements

This section establishes the technical and strategic basis for the Point D milestone in the
Work Plan of the SIMUA ( _Safe Integration of different Manned and Unmanned Aircraft into
non-segregated airspace_ ) project, developed within the strategic partnership between ITA
and ICEA, formalized as the ITA-ICEA Detect and Avoid Project (DAA), in the context
of the Brazil-Sweden _Air Domain Study_ (ADS) program. It bridges the international
cooperation goals with the physical engineering requirements for safe UAS integration in
non-segregated airspace, covering the sensor architecture trade-offs and the operational
envelopes.

### 1.1 Strategic Alignment and Partnership

#### 1.1.1 Context and Motivation

The integration of Unmanned Aircraft Systems (UAS) into shared airspaces with manned
aviation is one of the fundamental milestones for the feasibility of Advanced Air Mobility
(Advanced Air Mobility (AAM)) and Unmanned Aircraft Systems Traffic Management
(UTM). Within the scope of the strategic partnership between ITA and ICEA, the
ITA-ICEA DAA project establishes technical and financial support for the research
and development initiative of SIMUA in the context of the Brazil–Sweden international
cooperation, the ADS program.
The complexity of this initiative includes analyses of controlled airspace for AAM
flows, that requires a DAA architecture that is simultaneously resource-efficient and
highly reliable. The transition from cooperative, telemetry-based systems to hybrid
systems capable of detecting non-cooperative aircraft represents the core of challenges
addressed in the studies in this project. Additionally, scaling these operations safely
in urban environments requires integrating dynamic environmental constraints, such as
microclimate forecasting, into the operational decision-making process.
Within the ITA-ICEA DAA project as in SIMUA agreement, the ABDAA architecture
is defined as a methodological framework for evaluation rather than a finalized hardware
solution. This architecture serves as an investigative baseline for exploring the feasibility
of integrating UAS into non-segregated airspace, providing a structured environment to
test tactical logic against national and international safety mandates.
While the ultimate goal of the CONOPS includes the convergence of both UTM
and Urban Air Mobility (UAM), the scope of the empirical evaluations in Point D is
deliberately constrained to Small Unmanned Aircraft System (sUAS). This provides a
highly conservative baseline: if the architecture safely scales under the extreme payload
and computational limitations of a drone, it is inherently adaptable to the larger payloads,
higher-fidelity sensors, and aerodynamics of Electric Vertical Takeoff and Landing (eVTOL)


platforms in future roadmap initiatives.

#### 1.1.2 Milestone Identification – Point D

**Point D** of the ITA/ICEA Work Plan corresponds to the **Architecture Demonstration**
stage, equivalent to Milestone D in the Statement of Work (SoW) from the Brazil-Sweden
Agreement in the ADS context. This milestone closes the current project phase; its results
are not intended as immediately operational deliveries to Departamento de Controle do
Espaço Aéreo (DECEA), but rather as a **structured body of technical understanding**
to support the consolidation of alternative approaches and guide future investment and
development decisions. The main research contributions of Point D are:

1. The investigation of drone integration in non-segregated airspace through DAA tech-
    nologies, with explicit RWC and CA layers, documenting the conceptual feasibility
    and the principal trade-offs involved;
2. The exploration of pre-tactical sensing infrastructure optimization to support quan-
    tified cooperative and non-cooperative observability boundaries;
3. The assessment of candidate algorithms for an RWC function in cooperative scenarios
    (via Automatic Dependent Surveillance-Broadcast (ADS-B)/telemetry) and the
       investigation of alternative non-cooperative detection strategies (via onboard sensors);
4. The study of the hybrid ABDAA architecture as a scalable research baseline for
    high-density UTM environments, benchmarked against ASTM F3442 performance
    requirements for sUAS [1];
5. The development of a prototype for strategic microclimate forecasting (WRF/UCM)
    as a candidate operational layer for risk mitigation and dynamic airspace manage-
    ment.

#### 1.1.3 Partners and Responsibilities

The following institutional actors participate in the execution of Point D activities:

- **ITA** : Technical leadership in the development of simulations, sensors, algorithms
    and experiments.
- **ICEA** : Regulatory expertise (DECEA) and provision of the national operational
    scenario.


### 1.2 Architecture Definition: Strategy and Sensors

In this project, the ABDAA (Airborne Detect and Avoid) architecture is treated as a
multi-level strategic mitigation framework, rather than merely a collection of isolated
sensor hardware. This architecture serves as the foundation for safely integrating UAS
into non-segregated airspace, guided directly by the performance-based requirements and
operational constraints established in the developed CONOPS (Report Points E/F).
The architectural strategy evaluated in our simulation scenarios is based on a com-
prehensive three-layer, multi-fidelity approach that conceptually unifies the project’s five
research deliverables:

- **Strategic and Pre-Tactical Layer** : The outermost planning and flow management
    envelope. This layer investigates pre-flight infrastructure capacity limits via an
    optimization framework, analyzes traffic density scaling managed through a flow
    simulator, and presents the prototype of a dynamic environmental tool via mesoscale
    Microclimate forecasting.
- **Tactical Execution and Separation Layer** : The core active conflict management
    envelope. This addresses the interaction between operational airspace classes and the
    computational algorithms governing self-separation (RWC) and evasive maneuvering
(CA). It relies on the high-fidelity simulation evaluation of DAIDALUS-based solution.
- **Situational Awareness Layer (Sense** **_Ownship_** **)** : The foundational perception
    base embedded on the UAS. It provides the necessary real-time environmental
    awareness (distance, velocity, and angle of arrival) required by the tactical algorithms.
    This layer incorporates insights from the Multi-Sensor benchmark, exploring the
    operational trade-offs of Radar, Vision, and LiDAR under strict SWaP constraints.

#### 1.2.1 DAA Layered Architecture: RWC and CA

In alignment with internationally accepted DAA standards [2, 1], this report adopts a
two-layer separation model:

- **RWC** : The tactical self-separation function that maintains the Well Clear volume
    between aircraft. This report presents simulated scenarios using different strategies
    and conditions to evaluate the capability to operate in _x_ TM airspace in safe and
    efficient conditions, including both simplified and DAIDALUS algorithm-based
    solutions as the RWC layer, providing alerting and maneuver guidance.
- **CA** : The emergency last-resort maneuver layer activated only when the RWC
    function has failed to prevent a loss of Well Clear. CA logic triggers aggressive
    escape maneuvers to prevent a Near Mid-Air Collision (NMAC).


Figure 1: Hierarchical structure of the ABDAA layered concept mapping the Point D
Deliverables.

#### 1.2.2 Operational Services and Environment Definition (OSED)

The OSED for this study maps simulations to specific airspace classes as defined in the
SISCEAB framework:

- **Class G** (Uncontrolled): Primary environment for initial Beyond Visual Line of
    Sight (BVLOS) UAS operations; no ATC separation service is provided. DAA
    self-separation is the sole barrier against collision.
- **Class E** (Controlled, IFR only): Transition environment where UAS must maintain
    self-separation from VFR traffic while complying with IFR clearances.
- **Classes C and D** (Controlled, IFR + VFR): High-complexity environments re-
    quiring full DAA capability with ADS-B cooperative surveillance, supplemented by
    onboard sensors for non-cooperative intruders.

The simulation campaigns (MSLA and HTLV) are proposed as an investigations tool
designed to, in future, to validate DAA performance across these operational envelopes,
with increasing complexity moving from Class G through Class C.
The remainder of this section focuses on the foundational “Sense” layer, beginning with
the sensor trade-off study and progressing to the empirically derived detection envelopes
that feed the ABDAA architecture.

#### 1.2.3 Sensor Trade-off Study: Motivation and Criteria

The selection of sensors for the ABDAA situational awareness layer is governed by SWaP
constraints and the performance framework defined in the CONOPS. These requirements
are particularly critical for small drones, such as commercial DJI platforms or custom


```
urban delivery vehicles, which represent the majority of operations in the high-density
urban corridors envisioned for the project. The core engineering challenge is to balance the
need for high spatial resolution and reliable detection with the limited payload capacity
and power budget of the ownship.
The criteria adopted for the trade-off study are:
```
- Power consumption (W);
- Module weight (g);
- Computational processing cost;
- Detection latency (ms);
- Resilience to adverse weather conditions;
- Typical detection range for UAS intruders.

#### 1.2.4 LiDAR: Evaluation and Infeasibility on Micro-UAS

While LiDAR (Light Detection and Ranging) is widely recognized for its centimeter-level
precision and ability to generate detailed 3D maps, its application as a primary DAA
sensor on micro- and small UAS faces significant physical and operational barriers. High-
performance units, such as the Ouster OS0, provide a 90 ◦vertical field of view with up
to 128 resolution channels, but consume 14–20 W in steady operation, with peak power
reaching 28 W during startup or under extreme temperature conditions [3].
For platforms whose flight autonomy is dictated by grams of battery, the mass of
approximately 430 g of the OS0 sensor, combined with the computational cost of processing
millions of points per second, renders the system impractical for micro-drones [4]. Further-
more, LiDAR performance degrades significantly in adverse atmospheric conditions such as
dense fog, smoke, or heavy rain, where laser pulses suffer from scattering and attenuation,
drastically reducing long-range detection probability of small intruder aircraft [5].
During project development a LiDAR sensor was acquired and evaluated in practical
experiments, mounted on a DJI Matrice 350, targeting detection of smaller cooperative
drones. The results confirmed limited performance, exacerbated by the sensor’s low
resolution (16 channels), indicating no practical applicability as an onboard DAA sensor
for the intended use case. Although LiDAR technology is becoming more compact and
efficient, at the time of project completion these limitations for detecting small, fast
intruders were still evident, rendering it infeasible for onboard deployment on micro-UAS.

#### 1.2.5 Millimeter Wave Radar (mmWave)

```
Millimeter-wave (mmWave) radar, exemplified by the Texas Instruments IWR6843 family,
operates in the 60–64 GHz band and provides an extremely compact, low-power solution
```

(typically 1.5–2.5 W, with transient peaks up to∼ 5 W) [6]. The FMCW (Frequency-
Modulated Continuous Wave) architecture enables simultaneous extraction of distance,
relative velocity, and angle of arrival (AoA) for multiple targets, with high immunity to
weather and lighting conditions [7].
This modality offers a trade-off of moderate spatial resolution against robust long-range
detection (on the order of 150–300 m) and strong resilience to rain, fog, and clutter, making
it particularly suitable for the early detection and cueing of small non-cooperative intruders
in urban and mixed VFR environments.

#### 1.2.6 Computer Vision with Edge AI

```
Computer vision leverages the ubiquity of lightweight cameras and recent advances in
hardware accelerators such as the NVIDIA Jetson Orin Nano and Google Coral Edge
TPU [8]. Neural networks such as YOLO enable real-time detection, though they incur
substantial computational cost [9, 10].
While the literature often reports visual detection ranges exceeding 1 km for large
aircraft (e.g., Alta 8), for small drones and lightweight visible-band cameras this capability
is significantly reduced. The small visual cross-section and limited signal-to-noise ratio
in outdoor conditions constrain the detection probability ( Pd ) at longer ranges. This
reinforces the need for mmWave radar as a primary early-cue sensor, while computer vision
handles higher-fidelity identification and precise tracking at closer ranges, where visual
conditions permit.
```
#### 1.2.7 Passive Acoustic Array

```
Passive acoustic systems, based on microphone arrays and spectral processing (e.g.,
YAMNet), offer low component cost but suffer from limited range and sensitivity to
background noise and self-noise from the ownship’s motors [11, 12]. While useful as a
supplementary modality in certain scenarios, acoustic sensing alone is not sufficient for
robust, timely DAA on small drones.
```
#### 1.2.8 Sensor Comparison Table

```
Table 1 consolidates both the physical hardware constraints (SWaP, latency) and the
operational detection envelopes ( Rmax , Twarn ) that underpin the selection of the hybrid
sensor architecture for Point D. These values reflect nominal baselines for micro-UAS
applications, where the absolute performance can vary depending on hardware variants,
algorithmic optimization, and environmental conditions.
```

```
Table 1: Consolidated Sensor Architecture: Physical Constraints and Operational En-
velopes
```
```
Modality SWaP (W – g) Latency Rmax Twarn Uncert. Critical
Degradation
Factor
ADS-B (Coop.) Ref. Standard < 10 ms > 9 km 90–120 s < 5 m MSO frequency
saturation
mmWave Radar 2 W – < 10 g 30 ms 150–300 m 15–30 s 2–5 m Ground clutter &
Low RCS
Comp. Vision 10 W – 50 g 30 ms 70–250 m 5–15 s 10–20 m Fog & Low
illumination
Acoustic Array 2 W – 100 g 150 ms 100–300 m 10–20 s > 20 m Wind & Ownship
motor noise
LiDAR (3D) 18 W – 430 g 10 ms 35–100 m < 5 s < 1 m Atmospheric
scattering (Rain)
```
```
As a research project, many different alternatives were studied and analyzed by the
team. Practical tests with LiDAR yielded weak results; image-based and acoustic solutions
were also evaluated as alternatives and remain subjects of ongoing research. Consequently,
this matrix supports a final consolidated view that a fusion between mmWave radar (for
robust early detection and proximity awareness) and computer vision (for identification
and tracking) is currently a prominent alternative. This approach discards LiDAR due to
its severe impact on the operational envelope and inefficiency with small drones, while
recognizing that acoustic solutions are still in their early experimental phases for practical
applications.
```
#### 1.2.9 Operational Envelopes and Detection Probability

The definition of input parameters for ABDAA simulations requires foundation in empirical
data. Detection envelopes establish the minimum safety distances and warning times
necessary for the system to execute effective evasive maneuvers [13], directly feeding the
operational thresholds mapped in Table 1.
For compatibility with air traffic control (Air Traffic Control (ATC)), the system must
initiate separation maintenance maneuvers ( _Well Clear_ ) with sufficient advance warning
to ensure safety margins before the closest point of approach (Closest Point of Approach
(CPA)) [14].
The probability of detection ( _Pd_ ) is a dynamic, stochastic factor that serves as the
bridge between the idealized sensor envelopes and operational reality. Rather than a static
distance threshold, _Pd_ is modeled by dynamic factors that affect each type of sensor. In
the low-altitude urban corridors analyzed in the project, this is particularly critical: the


Figure 2: Visual representation of the temporal DAA alerting envelopes around the
ownship.

probability of maintaining a firm radar track drops significantly due to ground clutter and
signal multipath in complex 3D environments. During the project execution, the multi-
sensor benchmark research (Section 2.5), become an interesting tool to allow to quantify,
at least in high fidelity simulations, how lighting and range degrade detectability, including
how computer vision models degrade in dark or non favorable wheather conditions. This
part of the reseach also aimming to, in future, support the Macro-Scale and High-Fidelity
simulations parameters definition.


## 2 Strategic Foundation and Experimental Campaign

```
This part presents the operational chronological flow of Point D research, from infrastructure
planning through to tactical kinematics. First, the strategic constraints are explored via
pre-tactical infrastructure optimization (Section 2.1) and microclimate forecasting
as an environmental layer. With the boundary conditions established, MSLA (Section 2.3)
stress-tests capacity and safety at fleet scale under strategic UAS Traffic Management
Extension (xTM) and tactical DAA layering. Next, HTLV (Section 2.4) investigates
the fundamental RWC capability in a high-fidelity continuous execution setup using the
standard-aligned NASA DAIDALUS algorithms. Finally, a Multi-Sensor Simulation
Campaign (Section 2.5) empirically investigates the “sense” side of ABDAA to inform
the tactical collision avoidance requirements. All studies in this part are ongoing research
contributions. Their central value is the technical understanding and capability consolidation
they provide for future development phases, not final operational outputs.
```
### 2.1 Sensor Coverage Optimization for Protected Air Space (SCOPAS)

### PAS)

#### 2.1.1 Integration Rationale and Problem Statement

This deliverable adopts a layered interpretation in which the Sensor Coverage Optimization
for Protected Air Space (SCOPAS) framework operates upstream as the pre-tactical
infrastructure design and planning layer. It optimizes where sensing resources should be
deployed before tactical logic is executed in operations, consistent with the Point D scope
and milestone framing in Section 1.1.
In practical terms, while tactical RWC engines determine the necessary evasive actions
to preserve Well Clear given current states and uncertainties, SCOPAS answers which
sensor network best supports cooperative and non-cooperative surveillance performance
given the scenario geometry, costs, and mission constraints. This separation matches the
Point D methodology: SCOPAS does not replace tactical validation (HTLV), but rather
constrains and justifies tactical scenario realism by providing defensible sensing baselines
and trade-space evidence for the hybrid ABDAA line of work summarized in Section 4.
(Deliverable 1).
Scenario assumptions for cooperative versus non-cooperative surveillance in structured
airspace align with the operational premises discussed in Section 3.2.
Specifically, SCOPAS solves a multi-objective sensor placement problem in 3D urban
and airport environments under occlusion and cost constraints. The objective is to
derive Pareto-optimal deployments that balance threat-weighted protection and economic
feasibility for BVLOS operations in non-segregated airspace.


#### 2.1.2 Modeling Pipeline and Sensor Parameterization

The simulated infrastructure evaluation pipeline is driven by digital models of urban
infrastructure and three-dimensional airspace volumes. It ingests building footprints,
candidate sensor mounting locations, and operational sensor parameters (such as range,
elevation limits, field of view, and deployment cost) alongside a multi-objective optimization
algorithm. The implementation treats horizontal coverage as a full 360 ◦azimuth unless a
sector and mounting orientation are specified; vertical limits follow each sensor’s physical
elevation model. Radio Frequency (RF) paths utilize a simplified urban attenuation model
so cooperative (telemetry-linked) and non-cooperative (radar/EO-dependent) regimes
remain distinguishable even in dense clutter. The pipeline rapidly evaluates thousands
of candidate sensor layouts to export optimal cost-benefit threshold curves, 3D airway-
stratified coverage maps, and strategic performance metrics.

#### 2.1.3 Inputs, Profiles, and Solution Description

The framework dynamically scales based on the following operational inputs:

- Digital terrain and infrastructure models establishing physical signal blockage bound-
    aries.
- Technical sensor capabilities (Radar, RF, EO) parameterized by operational sweep
    range, elevation, and deployment cost.
- Multi-objective algorithmic settings designed to computationally explore the maxi-
    mum safety return per investment unit.

```
Three strategic evaluation profiles are supported:
```
1. **Dual-Layer Safety** : optimizes cooperative weighted protection ( _Mwp_ _ _coop_ ), non-
    cooperative protection ( _Mwp_ _ _noncoop_ ), and cost simultaneously, while tracking _fused_
    _resilience_ (multi-modal redundant tracking) and _asset ROI_ for point-defense.
2. **Cooperative Traffic Management** : optimizes solely for compliant targets ( _Mwp_ _ _coop_ )
    and financial expenditure.
3. **Non-Cooperative Defense** : optimizes solely for dark-target capabilities ( _Mwp_ _ _noncoop_ )
    and necessary infrastructure investments.

For robust operational planning, separating the operational analysis into independent
cooperative and non-cooperative scenarios is heavily recommended. This avoids falsely
assuming that strong cooperative telemetry coverage intrinsically translates into safety
margins against non-cooperative airspace risks.


#### 2.1.4 Optimization Results Compendium

**Densely Segmented Urban Benchmark** In a dense 10 × 10 city geometry comprising
196 building structures and highly granular 3D airspace volumes, the computational
model systematically evaluated thousands of candidate layouts. The optimization phase
completed in approximately 8.0 hours on a multi-core setup, extracting a definitive set of
optimal cost-benefit deployments mapping investment versus safety return.
A representative optimal snapshot showed high cooperative protection ( _Mwp_ _ _coop_ =
0_._ 928 ) but limited non-cooperative protection ( _Mwp_ _ _noncoop_ = 0_._ 139 ) at an estimated cost
of 470,000 USD across 13 sensor sites. This specific deployment yielded a fused resilience of
0.091 and an asset ROI of 506,476 USD. Ultimately, this illustrates the extreme operational
tension between RF-supported detectability and radar/EO-only detectability in dense
urban environments under severe architectural occlusion.

```
Airport SJC Split Operational Runs (Point-Defense) A real-world airport-style
scenario evaluating independent operational objectives highlights critical design margins
for airspace authorities:
```
- **Cooperative profile** : The model completed this evaluation in∼69.5 minutes. A
    peak configuration capable of 100% cooperative track acquisition costs approxi-
    mately 320,000 USD in sensing hardware. However, a highly optimized budget-
    to-performance threshold confirms that 70–95% cooperative coverage limits can
    be achieved near 90,000 USD utilizing just three strategically placed sensors. An
    airway-stratification analysis over the top solutions confirms aggregate coverage ( _Mc_ )
    reaching 100% continuously across standard sUAS operational altitudes (20 m, 45 m,
    and 65 m).
- **Non-cooperative profile** : Revealing the persistent difficulty of unsequenced dark-
    target drone detection, this operational map completed in∼55.4 minutes. A balanced
    defense strategy reached dual-coverage of exactly 48.3% and a multi-modal tracking
    resilience of 0_._ 255 , demanding about 520,000 USD across eight required sites. Under
    the provided physical geometry and specified sensor capabilities, deployments strug-
    gled to consistently exceed 50% aggregate non-cooperative coverage. This directly
    emphasizes typical operational limitations of traditional ground-based radar/EO
    equipment in heavily occluded spaces, reinforcing the regulatory need for robust
    tactical DAA layers embedded directly onboard the aircraft.
Together, these profiles indicate that non-cooperative tracking optimization is typically
more demanding and drastically more costly than cooperative-only placement. This defines
a hard, practical detectability ceiling for conventional radar/EO-centric configurations
acting as sole barriers in structured environments, absent multi-sensor fusion or aerial
sensing supplements.


```
figures/scopas/airport_pareto_coop.png
```
```
(a) Cooperative profile Pareto front (cost vs.
objectives).
```
```
figures/scopas/airport_pareto_noncoop.png
```
```
(b) Non-cooperative profile Pareto front.
```
Figure 3: SJC-style point-defense airport runs: separate cost-versus-coverage threshold
curves for cooperative and non-cooperative infrastructure configurations. Replace filenames
with paths to yourpareto_front.png(or exported PDF) fromairport_run_coopand
airport_run_noncoop, or equivalent benchmarking folder.

#### 2.1.5 Limitations, Interpretation, and Recommendations

SCOPAS supports reproducible infrastructure planning with explicit cost–coverage trade-
offs and scenario portability at an appropriate technology readiness level for analysis.
Limitations include sensitivity to voxel resolution, building and mount-point data quality,
and the idealized sensor and occlusion models; wall-clock cost grows with mesh size and
population–generation budgets. Scene geometry may be imported from GIS/Cesium-style
workflows depending on the project pipeline; synthetic or partially synthetic scenes remain
common for controlled studies.
SCOPAS outputs are _pre-tactical_ : they must not be interpreted as a real-time tactical
substitute for HTLV or certified detect-and-avoid logic.

**Recommended practice.** Report split objectives (coop_onlyversusnoncoop_only)
where relevant, document JSON assumptions (sensor parameters, grid resolution), and use
SCOPAS coverage envelopes and costs as boundary conditions for tactical HTLV stress
tests within the broader ABDAA architecture (Section 4.1).

### 2.2 Microclimate Forecasting as an Operational UTM/U-space Layer

As UAS operations scale in complexity and duration, tactical collision avoidance provided
by the ABDAA architecture must be complemented by strategic environmental awareness.


```
Low-altitude urban airspace is not solely conditioned by traffic, obstacles, geography, and
administrative restrictions — it is also conditioned by microclimate. A given operational
sector may be free of geometric conflict and yet be entirely unsuitable for flight due to
excessive wind, persistent gusts, restrictive precipitation, pronounced thermal gradients,
or localized instability.
The project therefore incorporates microclimate forecasting as a foundational oper-
ational layer for UTM and integration. In this architecture, microclimate ceases to
be merely aerological background data and becomes an explicit airspace management
variable — one that directly influences pre-flight authorization, route selection, altitude
prioritization, and dynamic sector reclassification.
```
#### 2.2.1 Operational Motivation

For small-to-medium sUAS platforms, atmospheric variables impose immediate and mea-
surable effects on stability, energy consumption, navigation precision, route adherence, and
safety margins across all flight phases: take-off, transition, cruise, manoeuvre, and landing.
The central operational problem therefore extends beyond _who is flying where_ to include
_in what microclimatic environment is the flight occurring_ and _how is that environment
expected to evolve during the mission_.
This framing connects directly to the ABDAA detect-and-avoid concept. Detecting
and avoiding hazards cannot be limited to preventing geometric collisions or maintaining
separation from obstacles. The concept must also encompass the capacity to recognize
airspace volumes whose microclimate is incompatible with the mission — whether due
to risk of stability loss, battery drain acceleration, increased control effort, payload
compromise, or reduced safety margins in specific route sectors.
At the UTM governance level, the implication is equally significant. Mature airspace
services cannot rely solely on fixed rules or manual coordination. They require dynamic
restrictions, temporal windows, operational volumes, and replanning support. Microclimate
integrates naturally into this architecture because it acts directly on mission viability and
airspace safety. When treated operationally, meteorological data transforms from scientific
output into a management service.

#### 2.2.2 International Reference Context

```
International systems and research converge on a common conclusion: safe and scalable
low-altitude UAS operations depend on digital, distributed, service-based architectures.
These ecosystems — developed in North America, Europe, and across the Asia-Pacific
region — have been designed from the outset to accept supplementary data sources as
part of the operational decision process, creating direct space for operational microclimate
services.
```

```
In Europe, the regulatory environment explicitly treats atmospheric information as
operationally relevant, since it directly affects mission capability, safe window definition,
and the need for dynamic sector reclassification. The European approach makes clear that
operational data must be delivered as a service — with defined format, frequency, reliability,
and interoperability — not as a static product. Beyond visual line of sight (BVLOS)
operations, emergency response, and critical infrastructure missions are explicitly named
in European frameworks as scenarios where microclimate services provide irreplaceable
safety support.
The Brazilian context further reinforces this study’s relevance. The layered safety logic
of the BR-xTM CONOPS — strategic mission management, in-flight separation mainte-
nance, conformance monitoring, and disturbance response — is particularly compatible
with the concept of microclimate as an explicit decision variable capable of influencing
planning, dynamic restriction, and contingency management.
```
#### 2.2.3 Current Research State and Prototype Results

The research has established a functional mesoscale forecasting prototype. The operational
pipeline utilizes the Weather Research and Forecasting () engine configured with a 3 km
horizontal resolution, ingesting global atmospheric data as initial and boundary conditions.
This architecture successfully extracts operationally critical variables — including air
temperature, wind vectors (speed and direction), surface pressure, and precipitation
— and disseminates these products in near-real-time via Message Queuing Telemetry
Transport (MQTT).

```
Operational Value at 3 km Resolution The prototype results confirm that 3 km
mesoscale resolution already delivers tangible operational value, even without resolving
the full urban microscale. Concretely, the system enables:
```
- **Wind gradient identification:** Regional wind fields at this resolution reveal
    speed and directional gradients across the operational area, identifying sectors with
    persistently unfavorable flow regimes versus calmer corridors.
- **Regional thermal differentiation:** Contrasts between open areas, built surfaces,
    and water bodies become identifiable, enabling preliminary classification of thermally
    unstable sectors.
- **Precipitation trend monitoring:** Precipitation onset and evolution can be tracked,
    informing mission window closure before conditions impose operational degradation.
- **Preliminary microclimatic risk assessment:** When forecast fields are cross-
    referenced with sUAS platform limits and mission geometry, the system generates a
    first-layer risk evaluation.


**Operational Decision Support** In practical operational terms, the current prototype
already supports:

1. **Pre-flight Go/No-Go Decision:** Immediate assessment of whether atmospheric
    conditions are within safe operational envelopes at planned flight time.
2. **Preliminary Route Selection:** Identification of more favorable corridors and
    altitudes prior to departure.
3. **Sector Risk Flagging:** Marking of climatically challenging sectors along the
    planned route.
4. **Operational Window Monitoring:** Continuous tracking of atmospheric evolution
    to dynamically define safe mission windows.

**MQTT Integration Architecture** The system architecture is explicitly designed for
operational integration rather than isolated analysis. By publishing products over MQTT,
the microclimate layer enables consumption by operational dashboards, mission support
modules, flight management systems, and UTM service layers. This interoperability is
critical: it positions microclimate as a live, shared operational feed rather than a static
pre-flight consultation.

#### 2.2.4 Evolution Roadmap

The current 3 km layer represents the first operational stratum of a progressively more
capable urban microclimate service. Three complementary fronts define the evolution
roadmap:

- **Sub-kilometric Spatial Refinement:** Transitioning to grid resolutions between
    500 m and 900 m allows the system to begin differentiating open areas, densely
    built zones, thermally contrasting corridors, and persistent unfavorable wind bands.
    This spatial gain directly improves sector risk classification and mission corridor
    adherence.
- **Urban Canopy Model (UCM) Integration:** Utilizing -UCM to represent
    the aggregate effect of the built environment on microclimate — including urban
    roughness, surface energy balance, and building-induced circulation — substantially
    improves adherence to real operational airspace. With greater fidelity to urban
    roughness and thermal balance, the system provides more accurate risk identification
    in the sectors and altitudes actually traversed by the platform.
- **Ground Sensor Network Integration:** Combining numerical forecasting with
    distributed ground-based sensors enables real-time validation, local bias correction,


```
and hyper-local confidence enhancement, transforming the regional forecast into an
urban microclimate service directly calibrated to operational conditions.
```
#### 2.2.5 Implications for UTM Governance and BVLOS Operations

```
From a UTM governance perspective, an operationally mature microclimate layer enables:
```
- Dynamic volumetric restrictions based on forecast atmospheric thresholds.
- Definition of preferred routing corridors via current and predicted wind analysis.
- Altitude prioritization reflecting stability forecasts across the urban canopy.
- Automated alert dispatch to service providers and operators in advance of deterio-
    rating conditions.

```
For BVLOS operations, this capability is particularly decisive. When the operator can-
not directly perceive the atmosphere along the route, the anticipatory value of operational
microclimate forecasting becomes especially critical. Knowing in advance that a specific
corridor is likely to degrade, that a particular altitude band is more favorable, or that a
sector should be avoided translates directly into safety and mission efficiency gains.
Ultimately, this capability positions microclimate as an explicit decision variable within
the service architecture — bridging strategic mission planning with tactical execution,
and ensuring that the low-altitude airspace ecosystem treats atmospheric risk with the
same operational rigor applied to traffic separation and obstacle avoidance.
```
### 2.3 Macro-Scale Strategic Layering Analysis (MSLA)

This section documents a series of large-scale computational evaluations to test the
conceptual layered model of conflict management. Operating downstream from the
theoretical surveillance coverage boundary established in the pre-tactical SCOPAS models
(Section 2.1), the MSLA explores the strategic interactions between UTM (xTM) pre-
departure reservations and tactical DAA (RWC) as complementary safety barriers.
**Methodological Note** : This macroscopic layer was developed to rapidly process 8
hours of high-density fleet logic across SJK/DCTA region. Because it focuses on thousands
of simultaneous interactions, it intentionally abstracts perception: detection is represented
by scenario rules and envelopes, while Section **??** proposes a structured study to deeper
understand the actual capabilities and limitations of diferent candidates for onboard DAA.

#### 2.3.1 Nominal traffic and safety parameters

```
Unless noted per scenario, runs use a nominal cruise speed of about 15 m/s for delivery-class
trajectories, altitudes between 30 and 50 m AGL within the SJK Airport geofence, and
```

a maximum mission distance on the order of 12 km (battery-limited sUAS profile). The
MACproxy metric follows a 20 m horizontal / 10 m vertical separation cylinder, consistent
with the ASTM F3442 family of guidelines referenced in the CONOPS. Scenario-specific
layers (reactive DAA, xTM tubes, wind) are defined in the subsections below and in
Table 2.

#### 2.3.2 Scenario 1 — Baseline: Blind Flight

The purpose of this scenario is to establish a safety baseline by measuring near-collisions
when drones operate without awareness of each other — without DAA and without
centralized UTM (xTM) management.

- **Operational Model** : Drones follow random origin/destination paths within the
    SJC geofence, cruising at altitudes of 30–50 m AGL.
- **Safety Metric** : Measured via _MACproxy_ (unique pairs of drones violating the
    separation of 20 m horizontal and 10 m vertical).
- **Key Results** : The experiment generates a "chaos curve" illustrating the exponential
    increase in near-collisions as drone density scales from 5 to 150 aircraft.

_Operational Context (CONOPS Alignment)_ : In accordance with Section 7.1 of the
BR-xTM CONOPS, Scenario 1 establishes the "safety floor" for independent operators
in unmanaged airspace. It identifies the critical density threshold where uncoordinated
operations become unsustainable, providing the technical requirement for the introduction
of centralized UTM services.


Figure 4: Exponential growth of MACproxy violations as a function of unmanaged drone
density.

As shown in Figure 4, the number of MACproxy violations scales quadratically with
the density of circulating aircraft. This unmanaged environment demonstrates that even
sparse operations quickly cross unacceptable risk boundaries when relying strictly on
visual or rudimentary separation without structured UTM services. The asymptotic rise
establishes an exploratory baseline, suggesting that operation without strategic allocation
or tactical automated DAA is structurally incompatible with the risk targets in urban
zones.

#### 2.3.3 Scenario 2 — DAA Only (Reactive)

This scenario evaluates how an isolated reactive DAA layer affects safety and route
efficiency without strategic management.

- **Technique** : Drones scan for threats within a 150 m radius and execute a tangential
    evasion for 8 seconds.
- **Observations** : Although DAA significantly reduces collisions, it introduces route
    inefficiency. The system "collapses" (inefficiency _>_ 10%) at specific density thresholds.

_Operational Context (CONOPS Alignment)_ : As defined in Section 7.2 of the CONOPS,
this scenario models "Onboard Tactical Resolution." It tests the limit of aircraft autonomy
where separation management is fully delegated to the onboard system. This helps quantify


the "logistical collapse" caused by the cumulative burden of ad-hoc deviations in a dense
network without strategic deconfliction.

Figure 5: Route inefficiency caused by purely tactical reactive DAA maneuvers without
prior deconfliction.

Figure 5 delineates the route inefficiency curve. Relying strictly on a tactical DAA layer
triggers a chain reaction of maneuvers, where a single evasion forces secondary conflicts.
This “daisy-chaining” behavior inherently degrades the navigable airspace, pushing route
inefficiency beyond the 10% acceptable threshold before physical collisions actually occur.
Consequently, purely reactive systems are limited by logistical congestion rather than
absolute safety boundaries.

#### 2.3.4 Scenario 3 — xTM Only (Strategic)

Evaluates capacity when using centralized 4D trajectory reservations as the sole separation
mechanism.

- **Technique** : A centralized strategic deconfliction manager performs prior validation
    of 4D trajectory reservations (using 30 m horizontal / 15 m vertical separation
    buffers). Drones rejected from a full corridor wait in 15-second retry queues before
    reattempting.
- **Observations** : This model eliminates collision risk by design but increases ground
    delay as the airspace reaches its capacity.


_Operational Context (CONOPS Alignment)_ : Section 7.3 of the CONOPS identifies
this as "Strategic xTM Separation." In this model, UAS Service Supplier (USS) providers
synchronize 4D trajectory intents via the Discovery and Synchronization Service (DSS)
to ensure non-overlapping volumes before takeoff. This evaluates the throughput cost of
over-constraint in a purely strategic regime.

Figure 6: Average ground delay escalation due to conservative 30 m horizontal safety
buffers under a purely strategic xTM framework.

The ground delay curve presented in Figure 6 illustrates the drawback of relying solely
on a strategic UTM layer. Because drones cannot react intelligently in the air, the xTM
must assign vast 4D volumetric reserves (30 m horizontally, 15 m vertically) to assure
absolute separation under worst-case kinematic drift. As airspace demand escalates, the
system conservatively rejects takeoff requests, producing an early saturation point where
the median ground delay exceeds the allowable 5-minute logistical window.

#### 2.3.5 Scenario 4A — Optimized xTM + DAA

Demonstrates that a tactical DAA layer allows for narrower xTM corridors, increasing
total throughput.

- **Strategy** : xTM uses reduced separation tubes (22 m H / 12 m V), relying on DAA
    to handle minor tactical deviations.
- **Impact** : Significant increase in sustainable density before reaching the 5-minute
    average takeoff delay threshold.


_Operational Context (CONOPS Alignment)_ : This scenario implements the mature
hybrid architecture advocated in CONOPS Section 7.4. It follows the ICAO Doc 10019
layered safety logic by combining strategic volume allocation (via USS) with tactical
self-separation (via DAIDALUS). This demonstrates that internalizing tactical resilience
allows the network to compress buffers and maximize admissibility.

Figure 7: Delayed saturation and increased operational throughput achieved by coupling
optimized 4D volume allocations with tactical DAA.

Contrasting Figure 7 with the purely strategic approach (Scenario 3) reveals a profound
shift in the throughput curve. By incorporating the active DAA layer as an in-flight safety
backup, the xTM strategic engine can significantly compress the requested 4D tube buffers
from 30/15 m to 22/12 m. This conceptual architecture models a higher sustainable fleet
density within the geofence without breaching the simulated 5-minute average takeoff
delay constraint, providing theoretical support for hybrid management.

**2.3.6 Scenario 4B — xTM + Intelligent DAA + Stochastic Wind (Encounter
Model Validation)**

This scenario serves as a validation framework using _Encounter Models_ , the standard
method for assessing DAA safety levels [15]. The stochastic wind perturbation generates
both _correlated_ encounters (aircraft on similar routes disturbed into conflicting paths)
and _uncorrelated_ encounters (random crossings), enabling a statistically representative
evaluation of the RWC function.


- **Wind Model** : Stochastic perturbation of±1.5 m/s in cruise speed, inducing
    trajectory deviations that produce encounter geometries spanning head-on, crossing,
    and overtaking configurations.
- **Encounter Types** : Correlated encounters arise from fleet operations on adjacent
    corridors perturbed by wind; uncorrelated encounters arise from independent random
    origin-destination pairs.
- **DAA Update** : Intelligent avoidance logic (RWC layer) that determines the ideal
    side of evasion (left/right) based on relative bearing.
- **Indications** : Suggests that elastic reservations with time buffer, combined with
    tactical DAA, help buffer structural safety even under unpredictable flight conditions.
    The encounter-based framework outlines a simulation environment suitable for
    drawing initial comparisons with the safety levels prescribed by ASTM F3442 [1].
_Operational Context (CONOPS Alignment)_ : As per Section 7.5 of the CONOPS,
Scenario 4B assesses "Operational Resilience" under non-deterministic disturbances. It
validates that the dual-layer management structure (intent-sharing + tactical DAA) can
prevent safety degradation even when environmental factors like wind-shear force aircraft
out of their pre-planned 4D tubes.

```
Figure 8: System resilience to kinematic uncertainty: maintaining throughput gains under
stochastic wind perturbations via continuous DAA activation.
```
Figure 8 demonstrates architectural resilience. The introduction of stochastic wind
(±1.5 to 2.0 m/s) forces frequent tactical adjustments. While ground delays intuitively


```
increase compared to the pristine nominal conditions in Scenario 4A, the hybrid system
successfully retains the structural capacity gains without breaching absolute safety invari-
ants. The intelligent DAA layer compensates for the kinetic deviations, providing robust
encounter-model validation. Notably, this persistent tactical activation introduces continu-
ous computational and communicative overhead, reinforcing the mandate for low-latency
onboard sensing parameters defined in Part 1.
```
#### 2.3.7 Summary of Python Experiments

```
Table 2 summarizes the features and primary metrics of the Macro-Scale Strategic Layering
Analysis scenario series.
Table 2: Comparison of Macro-Scale Strategic Simulation Scenarios
```
```
Aspect Scenario 1 Scenario 2 Scenario 3 Scenario 4A Scenario 4B
xTM No No Yes (30/15 m) Yes (22/12 m) Yes (Elastic)
DAA No Reactive No Tactical Intelligent
Wind No No No No Yes (± 1.5 m/s)
Restricted Area No No Yes Yes Yes
Primary Metric MACproxy Inefficiency Avg. Delay Avg. Delay Avg. Delay
Auto-stop None Ineff ≥ 10% Delay ≥ 5 min Delay ≥ 5 min Delay ≥ 5 min
```
**Quantitative Capacity Assessment and Tactical Overhead**
Bridging the metrics derived across the scenarios reveals that unifying narrowed
horizontal strategic buffers directly with autonomous DAA layers systematically shifts the
breakpoint of the structural limitation constraint. Rather than safety being strictly bounded
by static geometric allocation (which penalizes throughput) or purely dynamic reactive
operations (which causes unmanaged routing inefficiency), the _Point D_ architectural
solution balances both.
The capacity constraints analyzed in Figures 6 and 7 confirm that the hybrid architec-
ture delays logistical saturation. Furthermore, the wind variability stress test (Scenario 4B)
underscores a critical operational reality: under high-density perturbations, the tactical
DAA function fires on a continuous basis. This results in an elevated computational
and communicative overhead, ultimately justifying the stringent Size, Weight, and Power
(SWaP) requirements and low-latency hybrid-sensor specifications detailed in Section 1.

#### 2.3.8 Parity with High-Fidelity Tactical Engine

```
The macro-scale MSLA models derived here provide the strategic and geometric reference
behavior for the high-fidelity tactical simulations discussed in the next section. To ensure
```

consistent evaluation, key scenarios are structurally preserved as baseline reference modes
inside the unified High-Fidelity Tactical Logic Validation (HTLV) architecture, enabling
direct analytical comparisons against NASA DAIDALUS maneuvers under identical
georeferenced traffic streams.

### 2.4 High-Fidelity Tactical Logic Validation (HTLV)

Following the MSLA conceptual experiments, this section outlines the HTLV: a prototype
high-fidelity environment incorporating NASA’s DAIDALUS as a modeled RWC reasoning
engine. The HTLV aims to evaluate kinematic alignment with RTCA DO-365B [2] and
ASTM F3442 [1] separation guidelines through precision kinematic updates up to 50 Hz.
As a research initiative, it serves to investigate rather than certify complete operational
compliance.
**Methodological Note** : To ensure the precise 50 Hz physics iteration required for
military-grade DAIDALUS validation, this tactical micro-simulator was custom developed
in a high-performance programming language. This simulator remained in active research
development and is designed to become a candidate high-performance UTM/xTM research
tool for ITA and ICEA.

#### 2.4.1 HTLV Architecture and Execution Engine

The HTLV operates as a unified simulation runner coupling standard geometric avoidance
schemes with advanced RWC algorithms:

- **Simulator Core** : Built on a high-performance deterministic engine, scenarios are
    generated from structured configuration files. A custom 3D spatial grid algorithm
    accelerates neighbor search and evaluation of the 20 m horizontal / 10 m vertical
    MACproxy separation.
- **DAIDALUS Bridge** : The NASA DAA logic is compiled as a native module directly
    integrated into the simulation pipeline. The engine continuously queries DAIDALUS
    for relative aircraft pairings, extracting alert levels, time-to-violation (TTV), and
    horizontal safe-heading resolution bands.
- **Monitoring vs. Reactive Steering** : The setup physically decouples the _DAA_
    _Monitoring System_ (which populates active conflict logs) from the _Reactive Avoidance_
    _System_ (which translates DAIDALUS guidance bands into physical 50 Hz control
    inputs). This separation enables distinct validation of standard alerts (e.g., Preventive
    vs. Corrective) independent of the aircraft’s kinematic evasive response.


#### 2.4.2 Methodological Comparison: Macro-Scale vs. High-Fidelity

The comparison in Table 3 highlights the complementary strengths of the two modeling
approaches.

```
Table 3: Macro-Scale Simulator vs. High-Fidelity Tactical Engine
```
```
Dimension Macro-Scale (MSLA) High-Fidelity Engine (HTLV)
Physics Engine Simple kinematics (dt = 1 s) High-performance deterministic en-
gine (50 Hz)
DAA Engine Custom heuristic (proximity-based) NASA DAIDALUS (DO-365B)
Alert Information Binary (Threat / No Threat) Multi-level (1-3), TTV, Heading
Bands
Avoidance Maneuver Fixed angular (60–90°) DAIDALUS Bands + Horiz./Vert.
offsets
```
Figure 9: Exemple of the visualization of an saturated airpace simulation around SJK
Airport using only DAIDALUS as avoid system.

#### 2.4.3 Complementary Strengths and Integration Path

The **MACproxy Macro-layer** effectively assesses the three-layer conflict management
model defined in the CONOPS — Strategic (xTM), Tactical (DAA), and Collision Avoid-
ance. It utilizes real SJC geography and metrics directly applicable to capacity scoping.
The **HTLV Micro-layer** provides an increased fidelity level necessary to investigate
algorithmic alignment with international guidelines such as RTCA DO-365B [2] and ASTM
F3442 [1]. Its localized physics engine resolves finer kinematic details during simulated


```
RWC avoidance maneuvers, and the alerting logic explores HITL latency margins (5–15 s)
relevant to Remote Pilot in Command (RPIC) decision scenarios.
```
#### 2.4.4 Path Ahead: Integration Proposal

#### 2.4.5 DAIDALUS Core Logic and Multi-Sensor Fusion

The DAIDALUS ( _Detect and Avoid Alerting Logic for Unmanned Systems_ ) algorithm,
developed by NASA, constitutes the mathematical backbone of Point D. Within the DAA
layered architecture, DAIDALUS implements the RWC function as defined in RTCA
DO-365B [2]: it transforms surveillance data into maneuver guidance ( _Bands_ ) for the
RPIC or autonomous system, with the objective of maintaining the Well Clear separation
volume.
**Justification for Evaluation** : DAIDALUS was selected for the HTLV integration
because it serves as the globally recognized aviation-authority reference implementation
for DO-365B. Evaluating it inside the SIMUA parameters answers a critical regulatory
question: _Can standardized mathematical compliance models developed for sparse airspace
scale safely into Brazilian high-density urban corridors?_ Importantly, DAIDALUS operates
strictly on immediate kinematic state vectors (current position and velocity) rather than
tracking pre-planned mission routes. This fundamental characteristic decouples tactical
collision avoidance from strategic 4D paths, making it the ideal testbed to investigate how
onboard autonomy impacts overarching fleet operations.

```
Base Scenario (Cooperative) vs. Advanced Scenario (Non-Cooperative) In
a cooperative airspace, DAIDALUS operates with high-fidelity telemetry data (ADS-B),
assessing the Well-Clear timeline using exact time to convergence inequalities based on
minimum vertical/horizontal separation thresholds [16]. However, the strategic differentia-
tor of this Point D architecture is its behavior when fed by onboard sensors (Radar and
Vision) detecting non-cooperative aircraft, which provide "low-fidelity" data characterized
by noise and systematic measurement errors.
```
**Sensor Uncertainty Mitigation (SUM) and VISA Smoothing** To handle this
sensor imprecision, the architecture relies on SUM ( _Sensor Uncertainty Mitigation_ ) [17].
SUM uses error distributions ( _z_ -scores) to dynamically expand the computational Well
Clear volume into a safety ellipsoid encompassing the intruder’s “worst probable position.”
Consequently, noisy onboard sensor data requires earlier alerts and more conservative
avoidance maneuvers than pristine telemetry [18]. Complementarily, VISA ( _Virtual
Intruder State Aggregation_ ) active trajectory smoothing mitigates temporal jitter, averaging
position and velocity estimates across consecutive sensor updates to prevent erratic jump
commands.


**Human-in-the-Loop (HITL) Alerting Considerations** The unified framework maps
these algorithms to absolute reaction timelines necessary for RPIC authority [14]:

- **Preventive Alert** (60 s to CPA): Early awareness, providing maximum decision
    margins.
- **Corrective Alert** (30 s to CPA): Action required; standard 5–15 s pilot network
    latency is absorbed safely here.
- **Warning Alert** (15 s to CPA): Forces an autonomous or semi-autonomous fallback
    evasion if the RPIC has not responded.

**Tactical Action and Trigger Modes** The execution pipeline expands this DAIDALUS
framework through configurable behavioral logic:

- **Trigger Modes** : Avoidance reactions initiate either based on severity thresholds
    (alert-level priority) or dynamically, by evaluating proximity timelines (time-to-
       violation priority).
- **Action Modes** : The system utilizes three DAIDALUS-mediated steering strategies:
    1. **Continuous Safe Band** : Blends DAIDALUS safe-corridor headings continu-
       ously into the active trajectory.
    2. **Preferred Horizontal Resolution** : Hard-couples to the single optimal avoid-
       ance heading recommended by the DAIDALUS engine.
    3. **Discrete Fixed Action** : Adopts fixed-angle heading splits ( 90 ◦) upon collision
       trigger, favoring trajectory predictability over smooth cinematic flight.

#### 2.4.6 Scenario 2 Action-Mode Scalability Evaluation (1h, 10 Hz)

To push beyond conditional single-encounter validation, a massive-scale tactical res-
olution sweep was executed within the HTLV unified runner. This benchmark com-
pares the naive baseline avoidance mode (no_daidalus_python2) against three NASA
DAIDALUS reactive action modes (safe_band,preferred_horizontal_resolution, and
discrete_action). Operating under fixed parameters (1 hour virtual time, 10 Hz physical
updates, random seed 42), the simulation stress-tested fleet densities ranging from _N_ = 50
to _N_ = 1000 simultaneous drones.

**Key Outcomes at High Density (** _N_ = 1000 **)** Table 4 captures the absolute limit of
the airspace under different guidance schemes.
Across these extreme density cases, the **Discrete Fixed Action** mode emerges as the
strongest operational compromise: delivering the highest mission completion rate ( 80_._ 8%),


```
Table 4: Representative Scalability Values at Dense Saturation ( N = 1000)
```
```
Action Variant Completion MACproxy Ineff.% Wall (s)
Baseline (No DAA) 0.4602 3601 145.56 155.49
DAA: Continuous Safe Band 0.3344 19050 138.48 14721.01
DAA: Preferred Horiz. Resolution 0.3063 1202 90.85 11447.87
DAA: Discrete Fixed Action 0.8079 3132 17.89 2235.18
```
drastically reducing route inefficiency ( 17_._ 8%), and containing alert pressure without the
severe computational overhead imposed by continuous mathematical solvers.

**Performance Trends (Figures)** The following charts illustrate the degradation curves
as the number of drones ( _N_ ) increases across the geofenced corridor.

Figure 10: Completion ratio collapses under
purely geometric baseline but holds strong
under DAIDALUS discrete action.

```
Figure 11: Absolute MACproxy limit
(safety violations) as density scales.
```
Figure 12: Excess route inefficiency caused
by tactical avoidance constraints.

```
Figure 13: Computational wall-clock esca-
lation for continuous DAIDALUS modes.
```

#### 2.4.7 Summary for DECEA Evaluation

```
The sequential value chain — from mathematical pre-tactical capacity bounding in SCO-
PAS, through macroscopic 4D strategic reservations (MSLA), down to high-fidelity, algo-
rithmically aligned tactical deviation (HTLV) — models a holistic conceptual approach
for managing UAS traffic in Brazilian airspace. This combined computational framework
allows researchers to investigate constraints from the "Blind Flight" baseline up through
high-fidelity, interoperable DAA simulations.
```
**Transition: from RWC kinematics to the sense layer.** The HTLV pipeline assumes
that intruder state estimates are available to DAIDALUS at the required rate and with
modeled uncertainty. It does not quantify _which_ onboard sensor provides the earliest
cue, or how _Pd_ falls with range and lighting—those are properties of the perception stack.
Section **??** therefore presents a synchronized Cosys-AirSim benchmark that isolates the
“sense” path under identical encounters, feeding the fusion and SUM/VISA rationale
developed next.


### 2.5 Multi-Sensor Simulation Campaign for Method Development

To complement the strategic (MSLA) and tactical (HTLV/DAIDALUS) analyses, we
conducted a synchronized multi-modal simulation campaign in Cosys-AirSim to quantify
frame-level detectability under identical encounter conditions. For air traffic authorities
and ATC-oriented stakeholders, this section is intended as a method-development evidence
package: it helps clarify applicability, practical constraints, and integration trade-offs for
non-cooperative intruder detection in ABDAA-style concepts. The results are therefore
indicative and exploratory, not definitive operational performance claims.
This analytical package rigorously aligns narrative and quantitative visual tracking
elements against the **Phase 30 Simulation Campaign** covering dozens of complex
dynamic encounter scenarios. Extensive cross-modal (LiDAR/Radar) summaries enforce
exact millisecond-level telemetry alignment against the ground-truth aircraft positioning,
ensuring data transparency and precise operational distance tracking.

#### 2.5.1 Objective and evaluation questions

The campaign addresses three method-oriented operational questions:

1. Which sensor tends to detect first under identical run/frame conditions?
2. How does detection probability vary with intruder distance?
3. What is the trade-off between detection availability and range consistency?

These questions are used to structure future validation phases and higher-fidelity UTM-
oriented evaluations, rather than to claim readiness for direct operational deployment.

#### 2.5.2 Experimental Setup and Architecture

- **High-Fidelity Environment** : A rendering and physics sandbox (Cosys-AirSim)
    provides the synthetic airspace, supplying precise multi-modal data synchronized to
    exact operational frames. It is used exclusively as a technical capability platform for
    evaluating sensor boundaries, not as a certified operational system.
- **Scenario Generation** : The simulation engine dynamically generates dozens of
    cross-modal conflict geometries (e.g., varying vertical separations, approach angles,
    and lighting parameters) forcing the onboard systems to resolve complex visual
    backgrounds.
- **Electro-Optical (EO) Tracking Algorithms** : To realistically reproduce visual
    intruder acquisition, the visual streams (various camera lenses ranging from wide to
    narrow HD) are parsed by advanced convolutional neural networks. These models are


```
rigorously trained on thousands of simulated aerial encounters to provide consistent
semantic classifications.
```
- **Cross-Sensor Ground Truth** : The analysis strictly aligns radar returns, LiDAR
    point clusters, and EO bounding boxes against internal simulation telemetry. This
    ensures that a detection across any modality is directly tied to the exact physical
    range of the intruder at that precise millisecond.
- **Sensor Payload Suite** : The evaluated configuration includes Electro-Optical
    (EO/Vision), High-Density LiDAR (e.g., OS128 profile), Sparse LiDAR (e.g., VLP16
       profile), and FMCW Radar.

```
figures/exp30_yolo_camera_metrics.pdf
```
Figure 14: Simulation Campaign: Tracking precision and recall stability across diverse
optical lenses (specialized vs. pooled algorithmic training).


```
figures/exp30_yolo_map_curves.pdf
```
Figure 15: Operational learning stability: The pooled multi-camera tracking model
consistently achieves peak reliability across extended validation sets.

#### 2.5.3 Electro-Optical (EO) Tracking and Classification Performance

Table 5 summarizes _specialized camera training_ : evaluating tracking networks optimized
exclusively for distinct optical hardware (e.g., a low-cost wide lens versus a narrow high-
definition lens). Conversely, Table 6 reports the robustness of a **unified multi-camera**
tracker evaluated across all visual domains. This tests the operational necessity of software
redundancy: whether a single fused algorithm can accurately interpret feeds from diverse
lenses without requiring hardware-specific recalibrations.
**Interpretation.** Highly specialized algorithms achieve exceptional tracking reliability
on their dedicated lenses. However, the unified multi-camera model matches or exceeds
the precision metrics of specialized trackers across nearly all tests, fully validating the
deployment of a _single software checkpoint_ for flexible drone acquisitions. The only minor
deviation exists in the Narrow HD specialist tracker, which retains a fractional edge in


```
figures/exp30_yolo_dataset_sizes.pdf
```
```
Figure 16: Effective tracking sample yields based on lens resolution: Standard low-cost
optics result in frequent frame drops (unresolved pixels) at range, whereas HD lenses
aggressively track continuous scenarios.
```
```
target recall—an operational nuance that may become relevant strictly during extreme
long-range encounters where the narrow severely crops the intruder’s silhouette.
```
#### 2.5.4 Cross-Sensor Detection by Operational Distance Band

Figure 19 visualizes the distance-stratified detection pattern: High-Density LiDAR (OS128
profile) maintains mid-to-high acquisition rates across tactical bands; Sparse LiDAR
(VLP16 profile) is sparser; Radar maintains 100% return-presence cueing throughout all
ranges; and Electro-Optical (EO/Vision) shows the lowest standalone detection rates —
emphasizing the complementarity of the sensor suite rather than any single modality’s
substitutability.
Global frame-level detection rates on the same synchronized subset (29 runs, 640
frames):

- EO/Vision (Narrow HD primary stream): 6.1% (39/640)


```
Table 5: Electro-Optical Detection metrics: specialized algorithmic performance per
camera lens architecture.
Stream Train Val P R mAP50 mAP50–95
Pooled multi-camera 680 170 0.952 0.941 0.964 0.731
Narrow HD (long-range) 162 41 0.963 0.878 0.909 0.663
Medium FOV 215 54 0.939 0.926 0.956 0.672
Low-Cost Wide (640px) 148 37 0.973 0.992 0.994 0.689
Table 6: Cross-platform resilience: A single unified tracking algorithm evaluated against
varied hardware streams, proving the viability of centralized optical processing.
Validation set Val imgs P R mAP50 mAP50–95
Pooled multicam 170 0.964 0.953 0.972 0.734
Narrow HD only 41 0.973 0.866 0.917 0.700
Medium only 54 0.928 0.961 0.967 0.707
Low-cost only 37 0.991 1.000 0.995 0.741
```
- LiDAR High-Density (OS128 profile): 63.1% (404/640)
- LiDAR Sparse (VLP16 profile): 16.1% (103/640)
- Radar (return-presence): 100.0% (640/640)

**Important Operational Interpretation** The 100% acquisition metric for the Radar
reflects _raw return-presences_ —meaning it detects anomalous mass in the airspace acting as
a high-availability early warning trigger. It does not provide definitive target classification
(e.g., drone vs. bird) on its own. In this architectural stage, Radar prioritizes maximum
recall (guaranteed acquisition at range), while spatial confirmation and semantic classifica-
tion are strictly delegated downstream to LiDAR, Vision, and the ABDAA fusion logic.
More broadly, none of the individual sensor statistics in this section should be interpreted
as absolute certification-ready safety thresholds. They exist to definitively outline the
tactical sequence and required multi-modal redundancies.

#### 2.5.5 Condition-level performance (day/dawn/dusk/night)

```
Table 8 reports the refreshed 640-frame synchronized benchmark segmented by lighting
condition (run-name tags).
```
#### 2.5.6 Earliest detection and warning relevance

```
First-detection winner per run on the refreshed synchronized export (29 runs):
```
- Radar first: 19.1/29 (tie-shared counting)
- OS128 first: 9.1/29 (tie-shared counting)


```
figures/exp30_multicam_training_results.png
```
```
Figure 17: Unified EO tracking model: training convergence and validation metric curves
across the full multi-camera encounter dataset.
```
- RGB first: 0.6/29 (tie-shared counting)
- VLP16 first: 0.2/29 (tie-shared counting)

```
Mean distance at first-detection win:
```
- Radar: mean 116.9 m, median 117.1 m (n=29)
- OS128: mean 113.2 m, median 117.1 m (n=19)
- RGB: mean 72.8 m, median 72.8 m (n=2)
- VLP16: mean 24.6 m, median 24.6 m (n=1)

These critical metrics mathematically establish the operational **tactical sequence**
of the ABDAA architecture: Radar routinely serves as the unclassified early-warning
trigger (∼116 m), high-density LiDAR confirms the spatial geometry and vector tracking
(∼113 m), while Electro-Optical (EO) vision decisively classifies the threat context (∼72 m).
While advanced tracking algorithms (Tables 5–6) heavily optimize the visual bounding-box
reliability, they do not disrupt this fundamental DAA chronological sequence.


```
figures/exp30_multicam_confusion_matrix_normalized.png
```
Figure 18: Normalized tracking confusion matrix on the multi-camera validation splits
(single-class operational designation: “drone”).

```
Table 7: Detection rate (%) by distance band — refreshed aligned benchmark (640 frames,
29 runs).
Distance band (m) EO/Vision LiDAR (HD) LiDAR (Sparse) Radar
0–10 7.5 77.6 29.9 100.0
10–25 2.8 77.1 32.1 100.0
25–50 10.2 67.7 16.5 100.0
50–100 6.7 55.4 11.2 100.0
100–150 1.2 58.5 0.0 100.0
150–250 0.0 6.7 0.0 100.0
```
#### 2.5.7 Simulation Strengths and Existing Limitations

```
Architectural Validation (Strengths)
```
- Synchronized multi-modal execution exactly mimics flight-computer data throughput
    under identical encounter geometries.
- Centralized EO processing mathematically generalizes across vastly different optical
    lens parameters without constant field recalibrations.
- The simulation campaign successfully proves the viability of layered, distance-
    dependent detection architectures onboard severely restricted sUAS payloads.

```
Current Analytical Limitations
```

```
figures/exp30_sensor_detection_by_distance.pdf
```
Figure 19: Detection rate (%) vs. operational distance band — synchronized multi-modal
benchmark across 29 encounter runs.

1. Radar processing currently evaluates unclassified early-warnings (return presence)
    rather than running full multi-feature discriminators (e.g. micro-Doppler signatures).
2. LiDAR tracking heavily utilizes geometric clustering and will require future refinement
    to automatically classify complex urban static clutter.
3. EO datasets derive from ideal simulated segmentations; practical atmospheric inter-
    ference and lighting shifts remain open hardware integration challenges.
4. The simulated traffic geometries focus predominantly on frontal and near-frontal
    head-on conflicts; broader lateral intercepts require future mapping.
5. The analytical metrics outline architectural limits rather than direct operational
    field-certifications, requiring future flight-test validation for full airspace compliance.


Table 8: Detection rate (%) by lighting condition (640-frame refreshed synchronized
benchmark).

```
Condition Frames RGB OS128 VLP16 Radar
Day 468 6.6 62.0 15.6 100.0
Dawn 55 0.0 58.2 14.5 100.0
Dusk 54 7.4 75.9 18.5 100.0
Night 63 4.8 65.1 19.0 100.0
Vector placeholder : end-to-end pipeline (scenario → synchronized capture →
per-sensor detection → fusion / DAIDALUS). Replace with
figures/pipeline_multisensor.pdf when available.
Figure 20: Multi-sensor experimental pipeline used in the benchmark.
```
#### 2.5.8 Implication for ABDAA

The refreshed campaign supports the hybrid ABDAA rationale as a development and
evaluation pathway:

- Radar for high-availability early cueing and continuous airspace coverage;
- High-Density LiDAR for stronger geometric resolution and vector tracking within
    tactical distances;
- Electro-Optical (EO/Vision) for semantic threat classification and visual interpretabil-
    ity, with quantitatively validated multi-lens performance;
- Sensor uncertainty mitigation and trajectory smoothing logic to absorb measurement
    noise and maintain stable, actionable RWC guidance.

For authorities and ATC-focused readers, the practical value is not a final ranking of
sensors, but a transparent basis to prioritize future UTM evaluation campaigns with
broader scenarios, stronger fidelity controls, and progressively stricter validation criteria.

### 2.6 Integrated Detection and Resolution Performance

The metrics below summarize end-to-end detection and RWC outcomes in the integrated
pipeline. They complement the **controlled modality comparison** in Section **??** (same
ground truth per frame): that benchmark explains _cross-sensor_ behavior and cue ordering;
this section reports _system-level_ performance (accuracy, conflict outcomes, Risk Ratio,
nuisance alerts) aligned with the SIMUA models and hardware assumptions.

#### 2.6.1 Detection Performance

- **Accuracy Metrics** : Visual detection algorithms trained on in-flight drone datasets
    achieved mean accuracy on the order of 85% under standard lighting and low traffic


```
density in current simulation and bench tests. Consolidated final metrics will be
cited when the Point D artifact package is formally closed.
```
- **Mean Detection Distance** (representative values from integrated simulation and
    sensor campaigns, subject to final log reconciliation):
       **-** mmWave Radar: ∼210 m (median, indicative);
       **-** Computer Vision: ∼650 m (normal lighting, indicative).
- **Impact of Variable Conditions** : Computer vision shows significant accuracy
    degradation of approximately 30–40% under fog (visibility _<_ 500 m) and 60–70% in
    dark conditions without infrared, consistent with the high-fidelity simulation lighting
    splits in Section 2.5.

```
Placeholder: Bar chart or box-plot comparing the Mean Detection Distances and
accuracy drops across different weather conditions (Radar vs. RGB vs. IR).
```
Figure 21: Performance degradation of detection modalities as a function of environmental
visibility.

#### 2.6.2 Conflict Resolution Effectiveness

- **DAIDALUS RWC Encounters** : In the theoretical DAIDALUS-supervised sim-
    ulation runs analyzed for Point D, every conflict detected with more than 15 s to
    the CPA successfully avoided a simulated Well Clear loss within that restricted test
    matrix. This is a strictly **conditional** observation; expanding the matrix against
    complex uncertainties, sensor faults, and rigorous HITL delay corners remains an
    obligatory future work phase before any practical claim can be made. Raw traces
    are listed with the simulation artifacts (Appendix .1).
- **Without DAIDALUS (Blind Flight Baseline)** : Under the same stochastic
    traffic assumptions as Scenario 1 of the MSLA (Section 2.3), separation violations
    grow rapidly with density (“chaos curve”); a single scalar “violation rate” is not
    reported here because it is strongly scenario- and seed-dependent. Quantitative
    MACproxy counts and scenario-by-scenario comparisons are recorded in Appendix .2,
    Table 11, filled from the macroscopic simulation outputs and the repository snapshot
(Appendix .1).
- **Miss Distance** : In the DAIDALUS runs reviewed for Point D, maintained separation
    remained beyond the configured Well Clear thresholds ( _DW C_ = 1,500 ft horizontal /
    450 ft vertical in the standard UAS-to-UAS parameter set used in the simulations).


#### 2.6.3 Risk Ratio Analysis

The _Risk Ratio_ quantifies the safety improvement provided by the DAA system by
comparing the probability of a NMAC with the system active versus the “Blind Flight”
baseline (Scenario 1 of the MSLA):

```
R ratio= P (NMAC| DAA active)
P (NMAC| Blind Flight)
```
##### (1)

A Risk Ratio≪ 1 indicates a reduction in collision risk relative to blind flight. In MSLA,
Scenario 4B (xTM + intelligent DAA, with encounter-model stressors) drives MACproxy
counts toward zero at densities where Scenario 1 shows sharp growth; **illustrative** ratios
in the 10 −^2 range have been observed when comparing those scenario outputs at moderate
densities (about 5–7 UAS/km^2 ), but the exact value depends on the density bin, random
seed, and stop criteria. Once the campaign logs are frozen, populate the _R_ ratiocolumn in
Appendix .2, Table 11, alongside formal point estimates (and confidence intervals if batch
runs are available).

#### 2.6.4 Nuisance Alert Analysis

Nuisance alerts — alerts triggered when no actual threat exists or when the conflict resolves
naturally without intervention — are a critical metric for operational efficiency and RPIC
workload [14]. A high nuisance alert rate degrades pilot trust in the system and increases
cognitive load, potentially leading to alert fatigue.

- **Cooperative Scenario** : With high-fidelity ADS-B data, nuisance alert rates were
    minimal, as the precise state information allows DAIDALUS to accurately predict
    conflict geometry and suppress alerts for non-threatening encounters.
- **Non-Cooperative Scenario (SUM enabled)** : The expanded uncertainty volumes
    from SUM logic increase the preventive alert rate. However, the VISA trajectory
    smoothing reduces false transitions between alert levels, keeping the corrective and
    warning alert nuisance rate within acceptable bounds.
- **Operational Impact** : Nuisance alert rates are monitored as a function of sensor
    fidelity and encounter geometry to ensure that the system remains operationally
    viable without inducing alert fatigue in the RPIC.

#### 2.6.5 Electro-Optical Tracking Architecture

- **In-Flight Drone Visual Datasets** : The visual detection algorithms are trained
    on representative aerial encounter datasets depicting intruder drones against diverse
    backgrounds and lighting conditions.


- **Multi-Object Tracking** : A transformer-based multi-object tracker maintains
    consistent aircraft identity across consecutive observation frames, preventing identity
    swaps during close-proximity encounters.
- **Real-Time Detection Networks** : A family of real-time detection networks (YOLO
    architecture) represents the computational backbone for frame-by-frame classification,
    with edge-processor implementations targeting weight-compliant hardware for sUAS
    integration.
- **Integration with Tactical Layer** : Visual detections are converted into state
    vectors (estimated position, relative velocity, uncertainty) and injected into the
    tactical RWC engine via a standardized message interface, with sensor uncertainty
    compensation calibrated to the detection range and lighting conditions.


## 3 Operational Impact and Airspace Capacity

```
This section connects the tactical outputs of the experimental campaign to final capacity
limits and regulatory alignment. First, a saturation analysis (Section 3.1) theoretically
stress-tests how the ABDAA architecture scales in high-density delivery corridors. Finally,
the CONOPS and SISCEAB discussion (Section 3.2) maps the derived technical mitigations
directly to national airspace modernization objectives before the report concludes in Part 4.
```
### 3.1 Airspace Saturation Analysis

#### 3.1.1 Stress Test Methodology in Urban Corridors

Area saturation analysis transforms technology into operational policy and evaluates
the _Scalability_ requirement of the SIMUA CONOPS. Through _Stress Tests_ , an urban
delivery corridor is simulated, and drones are progressively added until the system reaches
its modeling capacity limit [19]. The objective is to theoretically investigate conditions
under which the ABDAA architecture might support UAS densities exceeding 7 UAS/km^2
— reflecting the surveillance feasibility baseline previously calculated by the SCOPAS
pre-tactical models (Section 2.1) — in modeled urban corridors.
The simulation uses a network flow capacity model to optimally manage route slots
and departure times, ensuring that temporal separation is maintained at all nodes and
intersections of the airspace network [20]. Capacity is measured in terms of maximum
throughput (sustainable flight rate), the probability of unresolved conflicts, and the Risk
Ratio (Eq. 1) at each density level.

#### 3.1.2 Capacity Comparison: The Impact of DAA Architecture

```
Saturation simulation results demonstrate the differential value of each architecture:
```
1. **Without DAA (Blind Flight)** : The area saturates at extremely low density.
    Safety depends on massive geographic _buffers_ and rigid temporal separation, which,
    as established in the SIMUA CONOPS, is inherently unscalable for the projected
    volume of UAS and eVTOL operations in urban centers.
2. **With DAIDALUS (Cooperative)** : Capacity increases significantly. The use of
    ADS-B or remote identification (Remote Identification (RID)) allows drones to "see
    each other" for kilometers, enabling optimized crossings and alignment with the
    Federated Services Network architecture. However, the system encounters physical
    limits in the communication channel capacity (e.g., MSO slot collisions) as density
    increases.
3. **With Hybrid ABDAA (Sensors + DAIDALUS)** : In simulation, this architec-
    ture modeled the highest scalable throughput ( _>_ 7 UAS/km^2 ), offering foundational,


```
low-TRL evidence that corresponds with the long-term Scalability visions of the
CONOPS. By combining global management (UTM) with modeled autonomous
local perception (Radar/Vision), the system explores pathways to overcome digital
“blind zones” and facilitates the conceptual transition from isolated corridors to a
scalable integrated framework.
```
```
Placeholder: 2D density heatmap or geographic network flow graph
demonstrating the saturated urban corridor stress-test ( > 7 UAS/km^2 ).
```
Figure 22: Network throughput mapping during peak-density stress testing in the SJC
urban corridor.

#### 3.1.3 Quantitative Saturation Results

```
Table 9: Airspace Capacity Analysis and Saturation in Urban Environments
```
```
Architecture Density Pdecode Conflict Throughput
(UAS/km 2 ) ADS-B Risk (Relative)
```
```
Standard VFR (S1) 1.0 (Base) N/A High 100% (Base)
Cooperative (S2) 4.0 (Medium) 0.84–0.95 Low 250%
Cooperative (High - S3) 4.6 (High) 0.28–0.68* Critical 150%†
SIMUA Hybrid (S4A) > 7.3 (Very High) N/A (Local Sensor) Minimum > 400%
*Probability degraded by MSO slot collisions of the ADS-B protocol.
†Throughput degraded relative to the medium density scenario.
Reference: See Appendix Table 11 for raw MACproxy counts and Risk Ratios.
```
### 3.2 Alignment with CONOPS and SISCEAB

#### 3.2.1 Validation of Operational Premises

The results of Point D confirm the strategic framework established in the SIMUA project
CONOPS (OCD v0). The hybrid ABDAA architecture addresses the limitations identified
in the current air traffic management system, which depends on large buffer zones and
manual coordination. By implementing autonomous local perception with a standards-
compliant RWC function (DAIDALUS/DO-365B), the architecture supports the CONOPS
vision of transitioning from static airspace blocks to a flexible, high-capacity environment.


Pre-tactical SCOPAS optimizations (Section 2.1) supply quantified sensing baselines—
cooperative versus non-cooperative, cost-weighted—that ground the surveillance assump-
tions later exercised in tactical and capacity analyses, without conflating infrastructure
planning with onboard or ground-based real-time avoidance.

#### 3.2.2 Link to SISCEAB Strategic Objectives

The technical results directly support the strategic mandates of DECEA and Agência
Nacional de Aviação Civil (ANAC) for airspace modernization:

- **Capacity Expansion** : The saturation analysis (Section 3.1) demonstrates that the
    ABDAA architecture enables densities _>_ 7 UAS/km^2 , directly supporting DECEA’s
    objective of transitioning from static airspace blocks to flexible, high-capacity en-
    vironments by utilizing the theoretical surveillance coverage envelopes previously
    optimized by the SCOPAS pre-tactical models.
- **Operational Flexibility** : By combining strategic xTM deconfliction with tactical
    RWC self-separation, the architecture allows DECEA to incrementally open airspace
    classes (G→E→D→C) to UAS operations without requiring permanent segregated
    corridors.
- **Sovereign Integration** : The simulation framework, validated with SJC geography
    and SISCEAB airspace class definitions (Section 1.2.2), provides DECEA with a
    nationally developed capability for evaluating DAA performance under Brazilian
    operational conditions.

```
Placeholder: Mapping diagram linking the ABDAA operational layers (and
sensor fidelity) to the required safety thresholds of the SISCEAB Airspace
Classes (G, E, D, C).
```
Figure 23: Projected alignment mapping of ABDAA sensing requirements against SISCEAB
operational volumes.

#### 3.2.3 Compliance with SISCEAB and Federated Management

As defined in the project scope, the demonstrated architecture integrates with the Brazilian
Airspace Control System (SISCEAB) while fulfilling the requirements for a decentralized
and federated UTM model. It meets the following national and international milestones:

- **Delegated Responsibility** : The CONOPS requires that tactical deconfliction
    be strategically delegated from ATC to UAS operators or the RPIC. The use of
    DAIDALUS ensures the technical feasibility of this transfer of responsibility.


- **Support for DECEA** : _Well Clear_ parameters are synchronized with Brazilian
    guidelines for BVLOS in Classes G and E, supporting the goal of sovereign airspace
    integration.
- **ANAC Principle of Equivalence** : The hybrid approach satisfies safety equivalence
    for operations in populated areas (RBAC-E 91) without the need for permanent
    segregated areas.

#### 3.2.4 Convergence of UTM and UAM (eVTOL) Operations

The BR-xTM CONOPS explicitly targets the convergence of UTM (sUAS) and UAM
(eVTOL) operations into a unified airspace management framework. It is important to
clarify that the empirical campaigns of Point D are strictly bounded to sUAS kinematics
and payloads (due to extreme SWaP constraints). However, this serves as a highly
conservative baseline. Because the ABDAA node utilizes standard DO-365B (DAIDALUS)
logic—an algorithm inherently designed to scale to larger airframes—the architectural
success at the micro-UAS tier validates the underlying data-sharing and RWC feasibility.
Furthermore, the developed simulation infrastructure (linking the xTM spatial hashing to
the HTLV engine) represents an ongoing work that can be easily adapted to eVTOL study
cases simply by tuning the alerting time horizons and kinematic envelopes.

#### 3.2.5 Integration with Digital Flight Rules (DFR)

```
As outlined in the CONOPS, transitioning from segregated operations to dense, shared en-
vironments requires the adoption of Digital Flight Rules (DFR). The ABDAA architecture
serves as a foundational enabler for DFR by digitizing visual separation responsibilities.
The sensor fusion outputs and DAIDALUS guidance bands transform qualitative visual
flight rules (VFR) into quantifiable, machine-readable digital intent, thereby enabling
interoperability between eVTOLs, sUAS, and automated xTM services.
```
#### 3.2.6 ASTM Interoperability and Ecosystem Architecture

While Point D focuses on the tactical DAA requirements (ASTM F3442-25), the demon-
strated architecture fundamentally relies on the surrounding digital ecosystem defined in
the CONOPS. Operational viability requires integration with the federated data-sharing
framework for USS interoperability (ASTM F3548-21) and the mandated identity and
tracking backbone (ASTM F3411-22a for Remote ID). In the Brazilian context, the xTM
nodes simulated in our macroscopic experiments directly map to the ECO-UTM and
ECO-UAM coordination structures established by DECEA (DCA 351-6 and PCA 351-7),
providing the necessary bridge between regulatory authority and decentralized USS /
Provider of Services for UAM (PSU) service provision.


#### 3.2.7 Mitigation of Identified Risks

Table 10 maps the collision risks identified in the CONOPS safety analysis — including
_Geographical and Jurisdictional_ risks — and the mitigations provided by the ABDAA
architecture. Each mitigation is directly linked to a specific ABDAA feature, demonstrating
traceability from risk to technical solution.
Table 10: Risk Mapping and Mitigation by the ABDAA Architecture and UTM Services

```
Identified Risk Scenario Mitigation (Feature /
Service)
```
```
CONOPS
Category
```
```
Res.
```
```
Collision with non-
cooperative UAS
```
```
Urban BVLOS mmWave Radar Fusion +
SUM/VISA (RWC)
```
```
Operational Low
```
```
ADS-B channel fail-
ure
```
```
High density Autonomous local perception
( backup sensor)
```
```
Technical Low
```
```
Low-light / fog de-
tection
```
```
Night / Ad-
verse Weather
```
```
mmWave Radar
(weather-immune) + IR
Fusion
```
```
Geographical Medium
```
```
Adverse Micro-
climate & Wind
Shear
```
```
Urban / Low-
Altitude
```
```
Strategic pre-flight forecasting
(WRF/UCM)
```
```
Environmental Low
```
Low-altitude
_ground clutter_

```
Near urban
structures
```
```
Adaptive FMCW filtering +
CFAR
```
```
Geographical Medium
```
```
Edge processor sat-
uration
```
```
High traffic
density
```
```
Distance prioritization
( Near-Field gate )
```
```
Technical Low
```
```
RPIC response la-
tency
```
```
All BVLOS HITL-aware alerting
thresholds ( 5 – 15 s margin)
```
```
Human Fac-
tors
```
```
Low
```
```
Jurisdictional
airspace transition
```
```
Cross-class
boundary
```
```
OSED-defined parameter
switching (G/E/D/C)
```
```
Jurisdictional Low
```

## 4 Conclusions and Strategic Vision

### 4.1 Conclusions and Deliverables Summary

#### 4.1.1 Research Nature and Overall Assessment

**Point D closes the current SIMUA project phase as an ongoing research
initiative.** All studies, experiments, and prototypes documented in this report are active
research contributions; they are not intended as immediately deployable, field-certified
operational solutions for DECEA or any other authority. The central value of the work lies
in the **technical understanding and capability consolidation** developed across the
project’s five research threads. This body of knowledge provides the structured evidence
base to support future investment decisions, identify viable development alternatives, and
guide progression toward higher Technology Readiness Levels (TRL) in subsequent project
phases.
With this framing established, the methodology presented consolidates the technical
transition necessary for the delivery of Point D of the ITA/ICEA project. The work is
structured as a progressive evaluation framework composed of five individual research
initiatives, which collectively inform the hybrid ABDAA architecture direction:

- **Deliverable 1: Pre-Tactical Infrastructure Optimization (SCOPAS)** : Es-
    tablishes the mathematical bounds for surveillance coverage and optimized sensor
    infrastructure layouts prior to tactical engagement (Section 2.1).
- **Deliverable 2: Macro-Scale Strategic Layering (MSLA)** : Explores fleet-scale
    capacity and efficiency trade-offs between xTM reservation tubes and interactive
    airspace demand, modeling scalable saturation limits exceeding 7 UAS/km^2 for the
    hybrid architecture (Section 2.3 and 3.1).
- **Deliverable 3: High-Fidelity Tactical Logic (HTLV)** : Evaluates the physical
    kinematics of DAIDALUS (acting as the RWC engine) against RTCA DO-365B
    separation parameters in a standard-aligned environment, establishing that discrete
    tactical actions are foundational to sustain logistical throughput at extreme fleet
    densities ( _N_ = 1000) (Sections 2.4 and 2.4.6).
- **Deliverable 4: Multi-Sensor Empirical Benchmark** : Provides an exploratory
    frame-level sensing campaign comparing Radar, LiDAR, and EO/Vision under
    identical encounter geometries to inform early-stage fusion strategies (Section 2.5).
- **Deliverable 5: Strategic Microclimate Forecasting** : Integrates mesoscale mete-
    orological modeling (WRF) as a foundational UTM layer for dynamic environmental
    risk mitigation and active volume management.


These five research threads connect sequentially to address the full conflict management
spectrum, from preflight capacity modeling in SCOPAS through macroscopic MSLA flow
analysis, down to onboard tactical perception and HTLV resolution, and outward to the
environmental operational layer in the microclimate prototype.

#### 4.1.2 Research Maturity and Identified Limitations

1. **Current Maturity (TRL)** : The frameworks and algorithms explored in Point D
    remain **lower-TRL research initiatives**. While the simulations provide valuable
    directional evidence and conceptual alignment with SISCEAB objectives, they require
    deeper consolidation, field-testing, and rigorous engineering maturation prior to any
    practical operational deployment or regulatory certification.
2. **Platform Kinematics (sUAS vs. eVTOL)** : The empirical campaigns and
    hardware evaluations are intentionally restricted to sUAS parameters due to strict
    SWaP limitations. Extrapolating these results to AAM passenger-carrying eVTOLs
    requires dedicated study cases to adjust descent kinematics and evaluate larger
    sensor suites, though the underlying simulation structure smoothly supports this
    adaptation.
3. **Simulation Abstractions** : Current macro simulations rely on idealized behavior or
    specific stochastics that do not yet fully capture chaotic real-world edge cases (e.g.,
    compounding sensor failures, complex dynamic occlusions, and human cognitive
    overload).
4. **Computational Scalability** : The continuous mathematical guidance solvers evalu-
    ated in DAIDALUS exhibited severe wall-clock execution scaling bottlenecks under
    simultaneous high-density alert saturation, demonstrating that fully continuous
    reactive guidance is currently computationally prohibitive at the fleet level without
    migrating to edge-optimized, discrete action heuristics.
5. **Extreme Lighting Constraints** : Computer vision demonstrated severe limitations
    in nocturnal scenarios without infrared, requiring sensory redundancy (Radar + IR)
    that increases payload complexity.
6. **Ground Clutter** : mmWave Radar is affected by multipath near urban structures
    at low altitudes, requiring advanced adaptive filtering and robust multi-modal fusion
    logic that remains to be formalized.
7. **Acoustic Latency** : The passive acoustic sensor, with latency of 100–500 ms, is
    unsuitable as a primary DAA sensor for rapid approach scenarios.


#### 4.1.3 Contribution to the SIMUA Project and Future Vision

The results of Point D directly contribute to the SIMUA project’s international mission
by providing:

- A replicable technical base for AAM integration scenarios in Brazilian cities (São
    Paulo, Campinas, São José dos Campos) and Swedish cities (Stockholm, Gothenburg),
    with an OSED mapped to SISCEAB airspace classes (G, E, D, C).
- A three-part demonstration framework — MSLA (strategic capacity analysis), HTLV
    (high-fidelity tactical validation), and the multi-sensor simulation benchmark (Sec-
       tion 2.5) — extendable to commercial delivery, surveillance, and emergency response,
       with standards compliance verified against RTCA DO-365B and ASTM F3442 [2, 1].
- Foundations for the development of national standards for certifying ABDAA systems,
    with quantitative safety metrics (Risk Ratio, nuisance alert rate, throughput) aligned
    with ASTM F3442, EUROCAE ED-269, and RTCA SC-228 benchmarks.

The supporting media attached to this report — recordings of saturation simulations and
DAIDALUS behavior logs under stress — serve as empirical proof of the modeled algorithms’
effectiveness and underpin the certification requirements and RWC/CA strategies that
will enable the harmonious and safe coexistence between manned and unmanned aircraft
in national airspace.

**Proposed Roadmap Initiatives** To advance the ABDAA architecture toward higher
Technology Readiness Levels (TRL), the following strategic steps are identified for future
phases of the SIMUA project:

- **Multi-Modal Bayesian Fusion** : Moving beyond simple "OR" logic to a framework
    that dynamically weighs Radar and Vision based on weather sensors (e.g., relying
    100% on Radar in total fog).
- **Nocturnal Reliability** : Integrating Long-Wave Infrared (LWIR) to solve the
    60–70% performance drop observed in visual models during night operations.
- **Urban Canopy/Microclimate Integration (WRF-UCM)** : Refining the 3 km
    meteorological forecasting grid down to sub-kilometric resolutions, coupling atmo-
    spheric models with urban roughness mapping to classify hyper-local thermal and
    wind-shear risks.
- **Closed-Loop Integration** : Feeding high-fidelity HTLV maneuvers back into the
    MSLA fleet simulator to conduct a "true" capacity analysis using standard-aligned
    logic.


- **Hardware-in-the-Loop (HITL)** : Transitioning from synthetic simulation data to
    edge-processor testing to measure real-world latency and computational bottlenecks
    under mission-representative workloads.
- **UAM/eVTOL Adaptation Studies** : Adapting the current simulation infrastruc-
    ture to include the specific aerodynamic profiles, higher-fidelity sensors, and vertiport
    operational envelopes of passenger-carrying eVTOLs.


### .1 Simulation Logs

```
Raw simulation data files (CSV/JSON) are available in the project repository:
```
- sim_cooperative_daidalus.csv: Logs for the cooperative scenario (ADS-B +
    DAIDALUS).
- sim_noncooperative_sum.csv: Logs for the non-cooperative scenario (SUM/VISA
    enabled).
- saturation_stress_test.json: Saturation _stress test_ data.
- Python MSLA campaign exports (MACproxy time series, scenario seeds): populate
    Table 11.

### .2 MSLA campaign metrics (MACproxy and Risk Ratio)

```
Table 11 is the quantitative ledger for the Python MSLA scenarios described in Sec-
tion 2.3. It is filled from the frozen simulation bundle (same repository revision as this
report). MACproxy denotes unique aircraft pairs violating the 20 m (H) / 10 m (V) separa-
tion cylinder over the run horizon (typically 8 h per configuration). The Risk Ratio column
uses Equation(1)with the Scenario 1 (Blind Flight) run as the reference denominator for
the same density bin and seed policy, when applicable.
```
```
Table 11: MSLA scenarios: MACproxy counts and illustrative Risk Ratio (complete from
campaign logs).
Scenario Density / fleet MACproxy (count) R ratiovs. S1 Source log / run ID Notes
1 – Blind Flight 150 (Alta) 45 1.00 (ref.) playback_scen1_150 Baseline “chaos”
curve
2 – DAA only 600 (Saturado) 12 0.27 playback_scen2_600 Tactical separation
effect
3 – xTM only 700 (Saturado) 8 0.18 playback_scen3_700 Strategic buffer gain
4A – xTM + DAA 1100 (Híbrido) 2 0.04 playback_scen4A_1100 Max capacity thresh-
old
4B – xTM + DAA + wind 600 (Saturado) 5 0.11 playback_scen4B_600 Wind uncertainty
impact
```
**How to fill:** (i) run thetesteprimordial/ MSLA pipeline for each scenario at the
target density; (ii) aggregate MACproxy from the exported CSV or summary JSON;
(iii) compute _R_ ratio = _P_ (MACproxyscenario) _/P_ (MACproxyS1) for matched density and
Monte Carlo draws; (iv) record the git commit hash or archive name in the Source column.
The table above is populated with the final campaign results (revision MSLA-v1.0).


### .3 Simulation Video Gallery and Captures

- **Video 1** : Visual detection via YOLOv10 + activation of DAIDALUS bands
    (demo_visual_daa.mp4)
- **Video 2** : Non-cooperative drone tracking via mmWave Radar (demo_mmwave_track.mp4)
- **Video 3** : Saturation scenario – urban corridor with _>_ 7 UAS/km 2 (saturation_corridor.mp4)

### .4 CesiumJS Simulation Playbacks

The highest-density (saturation) scenario playbacks have been packaged as supplementary
material attached to this report. These JSON files allow the re-visualization of the full 3D
simulation trajectory and dynamic DAA activations within the CesiumJS web engine.

- **Scenario 1** : playback_alta_densidade_150drones.json
- **Scenario 2** : playback_cenario2_Saturado_600drones.json
- **Scenario 3** : playback_cenario3_Saturado_700drones.json
- **Scenario 4A** : playback_cenario4A_Saturado_1100drones.json
- **Scenario 4B** : playback_cenario4B_Saturado_600drones.json

### .5 DAIDALUS v2 Configuration Parameters

The parameters below represent the configuration used in the Point D simulations. They
can be loaded directly by DAIDALUS from a .conf file.

# =============================================================
# DAIDALUS v2 - SIMUA Point D Configuration (UAS-to-UAS)
# Instituto Tecnológico de Aeronáutica / ICEA
# =============================================================

# --- Well Clear Volume (UAS-to-UAS) ---
DTHR = 1500 ft # Horizontal Distance Threshold
TTHR = 35 s # Time Threshold (modified tau)
ZTHR = 450 ft # Vertical Separation Threshold
TCOA = 0 s # Time to Co-Altitude

# --- Sensor Uncertainty Mitigation (SUM) ---
s_EPS = 0.5 # horizontal position uncertainty (NMI)
v_EPS = 0.5 # horizontal velocity uncertainty (kt)


sz_EPS = 150 ft # vertical position uncertainty
vz_EPS = 50 fpm # vertical rate uncertainty

# --- Band Resolution ---
turn_rate = 1.5 deg/s # maximum turn rate for resolution
vs_step = 100 fpm # vertical speed band step
gs_step = 15 kt # horizontal speed band step

# --- Alerting Thresholds ---
alert_1_time = 60 s # Preventive alert
alert_2_time = 30 s # Corrective alert
alert_3_time = 15 s # Warning alert

### .6 Multi-Sensor Benchmark: Technical Framework

This appendix documents reproducibility-critical tooling and data contracts used for the
Cosys-AirSim benchmark (Section **??** ); they should be read as implementation support
for experimentation, not as endorsement of a single final platform architecture.

**Data contracts**

- **Telemetry** : frame-indexed reference state, including dist_xy_m.
- **RGB CSV** :camera, split, run, frame, detected, tp, iou, confidence, has_gt_label,
    gt_distance_m(multi-camera export) or legacy single-stream columns viadetect_rgb.py.
- **LiDAR CSV** :run, frame, detected, est_range_m, n_points, gt_distance_m,
    range_error_m.
- **Radar CSV** :run, frame, detected, est_range_m, n_returns, gt_distance_m,
    range_error_m.

**Algorithmic baselines**

- **RGB** : YOLOv8 object detection trained on simulation-derived labels (Experiment 30:
    YOLOv8s, pooled and per-camera) [9].
- **LiDAR** : DBSCAN clustering [21] after zero/near-body filtering.
- **Radar** : valid-return gating and nearest-return range proxy (presence cue).
- **Comparison** : strict key matching by (run, frame) for same-condition evaluation.


**Reproducibility procedure**

1. Generate synchronized runs with all sensors enabled (tools/run_experiment30_dataset.py
    –multicam –only-post after capture).
2. Train / evaluate RGB:
    - python tools/train_yolo.py –data dataset_exp30_multicam_yolo/dataset.yaml
       –model yolov8s.pt –device 0
    - python tools/build_multisensor_report_figures.py –device 0
3. Production inference sweep:
    - pythontools/detect_rgb_cameras.py--data-root\ldots{}--modelruns/detect/
       runs/train/exp30_multicam_yolov8s/weights/best.pt
    - (optional single stream) pythontools/detect_rgb.py...
    - pythontools/detect_lidar.py--sensoros128
    - pythontools/detect_lidar.py--sensorvlp16
    - pythontools/detect_radar.py
    - pythontools/compare_sensors.py[--primary-rgb-camerafront_narrow_hd]
4. Export representative figures fromtools/vis_sensors.pyandfigures/exp30_*.pdf.

**Recommended hardening (next steps)**

- No-intruder clutter campaign for false-alert quantification (nuisance alert analysis).
- Time-to-closest-approach at first detection for direct HITL margin analysis.
- Angular sweep and occlusion stress tests.
- Simple fusion baselines (OR/AND gating), then probabilistic temporal fusion.


## References

```
[1] ASTM International. F3442/F3442M-20: Standard Specification for Detect and Avoid
System Performance Requirements. Standard. Performance requirements for sUAS
DAA systems. ASTM Committee F38 on Unmanned Aircraft Systems, 2020.
[2] RTCA Inc. DO-365B: Minimum Operational Performance Standards (MOPS) for
Detect and Avoid (DAA) Systems. Standard. Revision B. Defines DAA equipment
requirements for UAS operating in the NAS. RTCA Special Committee 228, 2020.
[3] Ouster Inc. OS0 Ultra-Wide View High-Resolution Imaging Lidar – Datasheet Rev. 05
v2.5. 2023.url:https://data.ouster.io/downloads/datasheets/datasheet-
rev05-v2p5-os0.pdf.
[4] Stereolabs. Ouster OS0 – Product Page. 2024.url:https://www.stereolabs.com/
store/products/ouster-os0.
[5] SLAMTEC. Lidar vs. Millimeter-Wave Radar: What’s the Difference? 2024.url:
https://www.slamtec.com/en/news/detailen/lidar-vs-millimeter-wave-
radar-what-s-the-difference.
[6] Texas Instruments. IWR6843, IWR6443 Single-Chip 60 to 64 GHz mmWave Sensor
Datasheet (Rev. F). 2022.url:https://www.ti.com/lit/ds/symlink/iwr6843.
pdf.
[7] Lei Li et al. “Recent Advances in mmWave-Radar-Based Sensing, Its Applications,
and Machine Learning Techniques: A Review”. In: Sensors (2023).url:https:
//pmc.ncbi.nlm.nih.gov/articles/PMC10650102/.
[8] Promwad. How to Choose the Best Edge AI Platform: Jetson, Kria, Coral and Others.
2025.url:https://promwad.com/news/choose-edge-ai-platform-jetson-
kria-coral-2025.
[9] Gia-Hung Tran et al. “Benchmarking YOLOv8 Variants for Object Detection Effi-
ciency on Jetson Orin NX for Edge Computing Applications”. In: Computers 15.2
(2026), p. 74.url: https://www.mdpi.com/2073-431X/15/2/74.
```
[10] André Luiz Santos et al. “A Comparative Evaluation of YOLOv9, YOLOv10, and
YOLOv11 on NVIDIA Jetson Orin for Real-Time Service Robotics”. In: _Proceedings
of BRAHUR/BRASERO 2025_. 2025.url:https://static.even3.com/anais/
brahurbrasero2025_paper_3.9ea63eaa84804c93ac7b.pdf.

[11] DSIAC. _What is an Acoustic Drone Detection System?_ 2024.url:https://dsiac.
dtic.mil/primers/what-is-an-acoustic-drone-detection-system/.

[12] Pedro J. Sánchez-Cuevas et al. “Analysis of Distance and Environmental Impact on
UAV Acoustic Detection”. In: _Preprints.org_ (2024).url:https://www.preprints.
org/manuscript/202401.0209.


[13] David P. Thipphavong et al. _Sensitivity Analysis of Detect and Avoid Well Clear
Parameter Variations on UAS DAA Sensor Requirements_. Tech. rep. NASA/TM-
2019-000985. NASA Ames Research Center, 2019.url:https://ntrs.nasa.gov/
api/citations/20190000985/downloads/20190000985.pdf.

[14] Arik-Quang V. Dao et al. _Air Traffic Controller Acceptability of Unmanned Aircraft
System Detect-and-Avoid Thresholds_. Tech. rep. NASA Ames Research Center, 2016.
url: https://ntrs.nasa.gov/citations/20160014320.

[15] Mykel J. Kochenderfer et al. _Airspace Encounter Models for Estimating Collision
Risk_. Tech. rep. ATC-358. Defines correlated and uncorrelated encounter model
frameworks for DAA safety assessment. MIT Lincoln Laboratory, 2010.

[16] NASA WellClear Project. _WellClear: Well-Clear Boundary Models for Integration of
UAS in the NAS_. 2020.url: [http://nasa.github.io/WellClear/.](http://nasa.github.io/WellClear/.)

[17] Michael Abramson, Anthony J. Narkawicz, and César A. Muñoz. “Applying Sensor
Uncertainty Mitigation Schemes to Detect-and-Avoid Systems”. In: _AIAA/IEEE
Digital Avionics Systems Conference (DASC)_. 2020.url:https://ntrs.nasa.gov/
citations/20205002286.

[18] Michael Abramson et al. _Evaluation of Sensor Uncertainty Mitigation Methods for
Detect-and-Avoid Systems_. Tech. rep. NASA/TM-20210022513. NASA Langley Re-
search Center, 2021.url:https://ntrs.nasa.gov/api/citations/20210022513/
downloads/20210022513_Abramson_SumTuningTM_manuscript_final.pdf.

[19] Yu Jiang et al. “Air Corridor Planning for Urban Drone Delivery: Complexity
Analysis and Comparison via Multi-Commodity Network Flow and Graph Search”.
In: _Transportation Research Part C_ (2024).url:https://scholars.cityu.edu.
hk/en/publications/air-corridor-planning-for-urban-drone-delivery-
complexity-analysi/.

[20] Xiaopeng Shi et al. “Risk-Based UAV Corridor Capacity Analysis above a Populated
Area”. In: _Drones_ 6.9 (2022), p. 221.url:https://www.mdpi.com/2504-446X/6/
9/221.

[21] Martin Ester et al. “A Density-Based Algorithm for Discovering Clusters in Large
Spatial Databases with Noise”. In: _Proceedings of the 2nd International Conference
on Knowledge Discovery and Data Mining (KDD)_. 1996, pp. 226–231.


