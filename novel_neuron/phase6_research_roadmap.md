# Phase 6: Realistic Research Roadmap

---

## Executive Summary

This roadmap provides a concrete, actionable plan for advancing oscillator reservoir computing (ORC) research. It is based on the corrected understanding that ORC is specialized for temporal pattern recognition at the edge, not general-purpose computing. The roadmap includes minimum viable experiments, go/no-go criteria, and realistic timelines.

---

## 1. Research Objectives

### 1.1 Primary Objective

**Demonstrate oscillator reservoir computing achieving competitive performance on keyword spotting task with <1 mW power consumption.**

### 1.2 Secondary Objectives

1. Validate theoretical bounds on memory capacity
2. Develop optimization framework for architecture design
3. Establish fair comparison methodology to digital baselines
4. Identify scaling limits and failure modes

---

## 2. Minimum Viable Experiment (MVE)

### 2.1 Experiment Specification

**Task:** Keyword spotting (10 words, 1 kHz audio)

**Architecture:**
- Oscillator type: Ring oscillator (CMOS-friendly)
- Array size: N = 16
- Topology: Small-world (Watts-Strogatz, k=4, p=0.1)
- Coupling strength: K = 0.1
- Readout: Phase detector + 8-bit ADC + linear classifier

**Implementation:**
- Platform: FPGA (Xilinx Artix-7 or similar)
- Oscillator emulation: Digital phase accumulators with noise injection
- ADC emulation: Quantization + noise model
- Training: Offline linear regression (ridge regression)

**Datasets:**
- Google Speech Commands (v0.02)
- Custom dataset: 10 words, 100 speakers, 10 utterances each

**Metrics:**
- Accuracy: Target >90%
- Power: Target <1 mW
- Latency: Target <100 ms
- Energy per detection: Target <50 μJ

### 2.2 Success Criteria

**Minimum Success:**
- Accuracy >85%
- Power <2 mW
- Latency <200 ms

**Target Success:**
- Accuracy >90%
- Power <1 mW
- Latency <100 ms

**Exceptional Success:**
- Accuracy >92%
- Power <500 μW
- Latency <50 ms

### 2.3 Timeline

**Month 1-2:** FPGA implementation of oscillator emulator
**Month 3:** Dataset preparation and baseline digital implementation
**Month 4:** Training and optimization
**Month 5:** Evaluation and comparison
**Month 6:** Analysis and reporting

---

## 3. Minimum Publishable Result (MPR)

### 3.1 Publication Targets

**Venues:**
- Primary: IEEE Journal on Emerging and Selected Topics in Circuits and Systems (JETCAS)
- Secondary: ISCAS (International Symposium on Circuits and Systems)
- Workshop: Neuromorphic Computing Workshop

### 3.2 Required Contributions

1. **Rigorous mathematical framework** (from Phase 2)
2. **Experimental validation** on real task (keyword spotting)
3. **Fair comparison** to digital baseline (TinyML)
4. **Theoretical bounds** validation (memory capacity, energy)
5. **Failure analysis** (noise, temperature, process variation)

### 3.3 Paper Structure

1. Introduction: Motivation and corrected positioning
2. Mathematical Framework: Theorems 1-7
3. Architecture Design: Optimization methodology
4. Experimental Setup: FPGA implementation, dataset
5. Results: Performance comparison, scaling analysis
6. Discussion: Limitations, failure modes, future work
7. Conclusion: Realistic assessment and positioning

### 3.4 Timeline

**Month 7-8:** Paper writing
**Month 9:** Submission
**Month 10-12:** Review and revision

---

## 4. Most Likely Failure Mode

### 4.1 Primary Failure Mode

**Phase noise prevents achieving target accuracy.**

**Why Likely:**
- Simulation showed 50% lock rate with N=4
- Phase noise limits effective precision to 4-6 bits
- Keyword spotting requires >90% accuracy

### 4.2 Mitigation Strategies

1. **Increase array size:** N=32 or N=64
2. **Stronger coupling:** K=0.5 or K=1.0
3. **Redundancy:** Multiple oscillators per input
4. **Averaging:** Longer observation time
5. **Calibration:** Per-device frequency trimming

### 4.3 Fallback Plan

If accuracy <85% after mitigation:
- Pivot to simpler task (binary wake word detection)
- Focus on theoretical contributions (mathematical framework)
- Publish as negative result (valuable for field)

---

## 5. Fastest Falsification Experiment

### 5.1 Experiment Design

**Goal:** Determine if phase noise fundamentally limits accuracy.

**Setup:**
- Simulate N=16 oscillator reservoir
- Vary phase noise level (S_φ from 10^-14 to 10^-10 rad²/Hz)
- Measure accuracy on simplified task (temporal XOR)

**Duration:** 1 week (simulation only)

### 5.2 Falsification Criteria

**If accuracy <70% at lowest noise level:**
- Phase noise is fundamental limitation
- Abandon high-accuracy applications
- Focus on theoretical work

**If accuracy >90% at lowest noise level:**
- Phase noise manageable
- Proceed with hardware implementation

### 5.3 Decision Point

After falsification experiment, decide:
- **Proceed:** If accuracy >90% at realistic noise
- **Pivot:** If accuracy <70% even at low noise
- **Optimize:** If accuracy 70-90% (try mitigation)

---

## 6. Lowest-Risk Prototype

### 6.1 Prototype Specification

**Platform:** Software simulation (Python)

**Components:**
- Kuramoto model simulator
- Phase noise injection
- Readout function (order parameters)
- Linear classifier (scikit-learn)

**Task:** Temporal XOR (simplified benchmark)

**Advantages:**
- Zero hardware cost
- Fast iteration
- Easy debugging
- Validates theoretical framework

### 6.2 Success Criteria

- Reproduce theoretical bounds (Theorems 1-7)
- Demonstrate >90% accuracy on temporal XOR
- Validate scaling laws (lock time vs N)

### 6.3 Timeline

**Week 1:** Implement simulator
**Week 2:** Validate theoretical bounds
**Week 3:** Scaling experiments
**Week 4:** Analysis and reporting

---

## 7. Required Fabrication Complexity

### 7.1 Silicon Implementation Options

**Option 1: Multi-Project Wafer (MPW)**
- Cost: $10k-$50k
- Timeline: 6-12 months
- Complexity: Low (standard CMOS)
- Risk: Low (mature process)

**Option 2: Full Custom Tapeout**
- Cost: $100k-$500k
- Timeline: 12-18 months
- Complexity: High (custom design)
- Risk: High (first-time design)

**Option 3: FPGA Prototype**
- Cost: $1k-$5k
- Timeline: 3-6 months
- Complexity: Low (digital emulation)
- Risk: Low (proven platform)

**Recommendation:** Start with FPGA (Option 3), progress to MPW (Option 1) if successful.

### 7.2 Process Node Selection

**28 nm CMOS:**
- Advantages: Mature, good RF performance, MPW available
- Disadvantages: Not cutting-edge
- Cost: Moderate

**65 nm CMOS:**
- Advantages: Very mature, low cost, MPW widely available
- Disadvantages: Larger area, higher power
- Cost: Low

**14 nm FinFET:**
- Advantages: Cutting-edge, low power
- Disadvantages: Expensive, limited MPW
- Cost: High

**Recommendation:** 65 nm for initial prototype (lowest risk, lowest cost).

---

## 8. Ferroelectric Necessity Analysis

### 8.1 Question: Are Ferroelectrics Necessary?

**Original ONC Claim:** Ferroelectric capacitors required for analog weight storage.

**Analysis:**

**Ferroelectric Advantages:**
- Non-volatile storage
- Analog tuning
- High density

**Ferroelectric Disadvantages:**
- Limited endurance (10^4-10^9 cycles)
- Process complexity
- Temperature sensitivity
- Linearity issues

**Alternative: Digital Weight Storage**

**Advantages:**
- Infinite endurance
- Perfect linearity
- Temperature stable
- Standard CMOS

**Disadvantages:**
- Requires DAC for analog coupling
- Higher area
- Static power

### 8.2 Recommendation

**Phase 1 (Initial Prototype):**
- **Do NOT use ferroelectrics**
- Use digital weight storage with DAC
- Simpler fabrication
- Lower risk
- Focus on oscillator dynamics

**Phase 2 (Optimization):**
- If successful, explore ferroelectrics for:
  - Non-volatile operation
  - Area reduction
  - Power reduction

**Rationale:** Ferroelectrics add complexity without being essential for proving the concept. They can be added later for optimization.

---

## 9. Go/No-Go Criteria

### 9.1 Technical Go/No-Go

**Go Criteria (must meet ALL):**
1. Accuracy >85% on keyword spotting
2. Power <2 mW
3. Phase noise manageable (accuracy degradation <10% with realistic noise)
4. Lock rate >80% across process variation

**No-Go Criteria (ANY):**
1. Accuracy <70% even with optimization
2. Power >5 mW (not competitive with digital)
3. Lock rate <50% (unreliable synchronization)
4. Calibration energy > computation energy

### 9.2 Economic Go/No-Go

**Go Criteria:**
- Energy advantage >2× vs digital baseline
- Cost advantage >1.5× vs digital solution
- Performance within 5% of digital baseline

**No-Go Criteria:**
- Energy disadvantage >2× vs digital
- Cost disadvantage >2× vs digital
- Performance disadvantage >10% vs digital

### 9.3 Decision Points

**Decision Point 1 (Month 2):** After software simulation
- If accuracy <70%: NO-GO (fundamental limitation)
- If accuracy >90%: PROCEED to FPGA

**Decision Point 2 (Month 6):** After FPGA prototype
- If power >5 mW: NO-GO (not competitive)
- If accuracy >85%: PROCEED to silicon

**Decision Point 3 (Month 12):** After silicon prototype
- If economic criteria not met: NO-GO (not viable product)
- If all criteria met: PROCEED to productization

---

## 10. Research Timeline

### 10.1 Year 1: Proof of Concept

**Q1 (Months 1-3):**
- Software simulation and theoretical validation
- Falsification experiment
- Lowest-risk prototype

**Q2 (Months 4-6):**
- FPGA implementation
- Dataset preparation
- Baseline digital implementation

**Q3 (Months 7-9):**
- Training and optimization
- Evaluation and comparison
- Paper writing

**Q4 (Months 10-12):**
- Paper submission
- Review and revision
- Decision on silicon implementation

### 10.2 Year 2: Silicon Validation (Conditional)

**Q1 (Months 13-15):**
- Chip design (if GO decision)
- MPW submission
- Test setup preparation

**Q2 (Months 16-18):**
- Fabrication
- Test board design
- Software development

**Q3 (Months 19-21):**
- Silicon testing
- Characterization
- Comparison to FPGA results

**Q4 (Months 22-24):**
- Final paper
- Productization assessment
- Go/no-go for commercialization

---

## 11. Resource Requirements

### 11.1 Personnel

**Year 1:**
- 1 PhD student (full-time)
- 1 PI (20% time)
- 1 Engineer (part-time, FPGA)

**Year 2 (conditional):**
- 1 PhD student (full-time)
- 1 PI (20% time)
- 1 Engineer (full-time, silicon)

### 11.2 Equipment

**Year 1:**
- FPGA development board: $5k
- Test equipment (oscilloscope, spectrum analyzer): $20k
- Computing resources: $5k

**Year 2 (conditional):**
- MPW cost: $20k
- Test equipment upgrade: $10k
- Probe station: $15k

### 11.3 Budget

**Year 1:** $100k
- Personnel: $60k
- Equipment: $30k
- Travel/overhead: $10k

**Year 2 (conditional):** $200k
- Personnel: $100k
- MPW: $20k
- Equipment: $50k
- Travel/overhead: $30k

**Total (2 years):** $300k (conditional on Year 1 success)

---

## 12. Risk Assessment

### 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Phase noise limits accuracy | High | High | Early falsification experiment |
| Synchronization unreliable | High | High | Redundancy, stronger coupling |
| Power budget exceeded | Medium | High | Optimize architecture |
| Calibration overhead | Medium | Medium | One-time calibration |

### 12.2 Economic Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Digital costs decrease | High | Medium | Focus on niche applications |
| Market too small | Medium | High | Early market validation |
| Competitors emerge | Medium | Medium | IP protection, speed to market |

### 12.3 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Timeline slip | Medium | Medium | Conservative estimates |
| Personnel turnover | Low | High | Documentation, redundancy |
| Funding shortfall | Medium | High | Phased funding, go/no-go points |

---

## 13. Success Metrics

### 13.1 Technical Success

- **Minimum:** Publishable paper with >85% accuracy
- **Target:** Paper with >90% accuracy, <1 mW power
- **Exceptional:** Paper with >92% accuracy, <500 μW power, silicon prototype

### 13.2 Academic Success

- **Minimum:** 1 conference paper
- **Target:** 1 journal paper + 1 conference paper
- **Exceptional:** 2 journal papers + citations

### 13.3 Economic Success

- **Minimum:** Clear advantage in specific niche
- **Target:** 2× energy advantage vs digital
- **Exceptional:** Startup formation or licensing

---

## 14. Conclusion

This roadmap provides a realistic, phased approach to advancing oscillator reservoir computing research. Key features:

1. **Go/no-go criteria** at multiple decision points
2. **Low-risk initial prototype** (software simulation)
3. **Clear success metrics** for each phase
4. **Contingency plans** for likely failure modes
5. **Realistic budget and timeline**

The roadmap is designed to:
- Minimize risk through early falsification
- Enable clear go/no-go decisions
- Provide publishable results even if hardware fails
- Scale investment based on demonstrated success

**Critical Insight:** This roadmap is fundamentally different from the original ONC vision. It focuses on a specific, realistic use case (temporal pattern recognition at the edge) rather than claiming general-purpose superiority. This is the only path to scientifically defensible and potentially impactful research.
