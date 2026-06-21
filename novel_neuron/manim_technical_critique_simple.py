"""
Manim Video: The Discovery - Technical Critique of Oscillatory Neural Computation
===============================================================================

Simplified version using only Text (no LaTeX required).

Run: manim -pqh manim_technical_critique_simple.py TechnicalCritiqueVideo
"""

from manim import *


class TechnicalCritiqueVideo(Scene):
    """Main video scene covering the complete technical critique discovery."""
    
    def construct(self):
        # Part 1: Introduction
        self.part1_introduction()
        self.wait(1)
        
        # Part 2: Original ONC Claims
        self.part2_original_claims()
        self.wait(1)
        
        # Part 3: Literature Survey
        self.part3_literature_survey()
        self.wait(1)
        
        # Part 4: Mathematical Corrections
        self.part4_mathematical_corrections()
        self.wait(1)
        
        # Part 5: Simulation Results
        self.part5_simulation_results()
        self.wait(1)
        
        # Part 6: Corrected Framework
        self.part6_corrected_framework()
        self.wait(1)
        
        # Part 7: True Use Case
        self.part7_true_use_case()
        self.wait(1)
        
        # Part 8: Research Roadmap
        self.part8_research_roadmap()
        self.wait(1)
        
        # Part 9: Conclusion
        self.part9_conclusion()
    
    def part1_introduction(self):
        """Introduction to the technical critique."""
        title = Text("The Discovery", font_size=72).to_edge(UP)
        subtitle = Text("Technical Critique of Oscillatory Neural Computation", font_size=36).next_to(title, DOWN)
        
        self.play(Write(title), run_time=2)
        self.play(Write(subtitle), run_time=2)
        self.wait(2)
        
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Problem statement
        problem = Text("The Problem:", font_size=48, color=YELLOW).to_edge(UP)
        self.play(Write(problem))
        self.wait(1)
        
        claims = VGroup(
            Text("O(1) summation via injection locking", font_size=32),
            Text("10^12 ferroelectric endurance", font_size=32),
            Text("Universal Adler equation applicability", font_size=32),
            Text("8x higher information density", font_size=32),
            Text("General-purpose GPU replacement", font_size=32)
        ).arrange(DOWN).next_to(problem, DOWN, buff=1)
        
        for claim in claims:
            self.play(Write(claim), run_time=0.5)
            self.wait(0.3)
        
        self.wait(2)
        self.play(FadeOut(problem), FadeOut(claims))
    
    def part2_original_claims(self):
        """Detailed breakdown of original ONC claims and why they were wrong."""
        title = Text("Original Claims vs Reality", font_size=48, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create comparison using text
        comparison = VGroup(
            Text("Original Claim -> Reality -> Severity", font_size=24, color=YELLOW),
            Text("O(1) summation -> Lock time ~ N^a -> FATAL", font_size=20),
            Text("10^12 endurance -> 10^4-10^9 cycles -> HIGH", font_size=20),
            Text("Universal Adler -> Fails for ring -> HIGH", font_size=20),
            Text("8x information density -> Ignores time -> HIGH", font_size=20),
            Text("GPU replacement -> Temporal only -> HIGH", font_size=20)
        ).arrange(DOWN).next_to(title, DOWN, buff=1)
        
        self.play(Write(comparison))
        self.wait(3)
        
        self.play(FadeOut(title), FadeOut(comparison))
        
        # Highlight the most critical error
        critical = Text("Most Critical Error: O(1) Summation", font_size=48, color=RED).to_edge(UP)
        explanation = VGroup(
            Text("Original: tau_lock ~ constant (coupling scales with N)", font_size=28),
            Text("Reality: tau_lock ~ N^a, a in [0.5, 1.0]", font_size=28),
            Text("Why: Parasitic coupling reduces effective K", font_size=28),
            Text("Evidence: Simulation showed 0% locking at N>4", font_size=28)
        ).arrange(DOWN).next_to(critical, DOWN, buff=1)
        
        self.play(Write(critical))
        for line in explanation:
            self.play(Write(line), run_time=0.5)
            self.wait(0.3)
        
        self.wait(2)
        self.play(FadeOut(critical), FadeOut(explanation))
    
    def part3_literature_survey(self):
        """Literature survey findings."""
        title = Text("Literature Survey", font_size=48, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Key papers
        papers = VGroup(
            Text("1. Torrejon et al. (Nature 2017)", font_size=32, color=GREEN),
            Text("Spintronic oscillator reservoir computing", font_size=24),
            Text("Spoken digit recognition achieved", font_size=24),
            Text("Uses RESERVOIR COMPUTING, not direct neural", font_size=24, color=YELLOW),
            
            Text("2. Chiba et al. (2024)", font_size=32, color=GREEN),
            Text("Reservoir computing with Kuramoto model", font_size=24),
            Text("Universal approximation theorem proved", font_size=24),
            Text("Output = linear combination of order parameters", font_size=24, color=YELLOW),
            
            Text("3. Hong & Hajimiri (JSSC 2019)", font_size=32, color=GREEN),
            Text("General theory of injection locking", font_size=24),
            Text("Adler equation FAILS for ring oscillators", font_size=24, color=RED),
            
            Text("4. Francois et al. (2024)", font_size=32, color=GREEN),
            Text("HfO2 ferroelectric capacitors", font_size=24),
            Text("Endurance: 10^4-10^9 cycles (NOT 10^12)", font_size=24, color=RED)
        ).arrange(DOWN).next_to(title, DOWN, buff=1)
        
        for paper in papers:
            self.play(Write(paper), run_time=0.5)
            self.wait(0.2)
        
        self.wait(2)
        
        # Key insight
        insight = Text("Key Insight: All experimental work uses RESERVOIR COMPUTING", 
                     font_size=36, color=YELLOW).to_edge(DOWN)
        self.play(Write(insight))
        self.wait(2)
        
        self.play(FadeOut(title), FadeOut(papers), FadeOut(insight))
    
    def part4_mathematical_corrections(self):
        """Mathematical framework corrections."""
        title = Text("Mathematical Corrections", font_size=48, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Theorem 1: Memory Capacity
        theorem1 = Text("Theorem 1: Memory Capacity Bound", font_size=36, color=GREEN).to_edge(UP)
        theorem1.shift(DOWN)
        self.play(Write(theorem1))
        
        eq1 = Text("MC <= N * K / (K + gamma)", font_size=32).next_to(theorem1, DOWN, buff=1)
        self.play(Write(eq1))
        self.wait(1)
        
        explanation1 = VGroup(
            Text("N: Number of oscillators", font_size=24),
            Text("K: Coupling strength", font_size=24),
            Text("gamma: Phase noise strength", font_size=24),
            Text("Phase noise fundamentally limits memory", font_size=24, color=YELLOW)
        ).arrange(DOWN).next_to(eq1, DOWN, buff=1)
        
        for line in explanation1:
            self.play(Write(line), run_time=0.3)
            self.wait(0.2)
        
        self.wait(2)
        self.play(FadeOut(theorem1), FadeOut(eq1), FadeOut(explanation1))
        
        # Theorem 4: Energy Efficiency
        theorem4 = Text("Theorem 4: Energy-Efficiency Bound", font_size=36, color=GREEN).to_edge(UP)
        theorem4.shift(DOWN)
        self.play(Write(theorem4))
        
        eq4 = Text("E_op >= N * P_osc / f_osc", font_size=32).next_to(theorem4, DOWN, buff=1)
        self.play(Write(eq4))
        self.wait(1)
        
        explanation4 = VGroup(
            Text("Static power cannot be amortized indefinitely", font_size=24),
            Text("Low utilization -> static power dominates", font_size=24, color=YELLOW),
            Text("Original 'amortized to zero' claim was WRONG", font_size=24, color=RED)
        ).arrange(DOWN).next_to(eq4, DOWN, buff=1)
        
        for line in explanation4:
            self.play(Write(line), run_time=0.3)
            self.wait(0.2)
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(theorem4), FadeOut(eq4), FadeOut(explanation4))
        
        # Observation time constraint
        obs_time = Text("Observation Time Constraint", font_size=36, color=GREEN).to_edge(UP)
        obs_time.shift(DOWN)
        self.play(Write(obs_time))
        
        eq_obs = Text("tau >= 2^N / f_osc", font_size=32).next_to(obs_time, DOWN, buff=1)
        self.play(Write(eq_obs))
        self.wait(1)
        
        explanation_obs = VGroup(
            Text("For N-bit precision, need tau >= 2^N / f_osc", font_size=24),
            Text("Example: 8-bit precision at 1 GHz -> tau >= 256 ns", font_size=24),
            Text("High precision fundamentally limits throughput", font_size=24, color=YELLOW),
            Text("Original '8x information density' ignored this", font_size=24, color=RED)
        ).arrange(DOWN).next_to(eq_obs, DOWN, buff=1)
        
        for line in explanation_obs:
            self.play(Write(line), run_time=0.3)
            self.wait(0.2)
        
        self.wait(2)
        self.play(FadeOut(obs_time), FadeOut(eq_obs), FadeOut(explanation_obs))
    
    def part5_simulation_results(self):
        """Simulation results."""
        title = Text("Simulation Results", font_size=48, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Monte Carlo results
        mc_title = Text("Monte Carlo Simulation (N=4, 10 runs)", font_size=32).next_to(title, DOWN, buff=1)
        self.play(Write(mc_title))
        
        results = VGroup(
            Text("Lock rate: 50%", font_size=28, color=RED),
            Text("Mean lock time: 391 ns (high variance)", font_size=28),
            Text("Order parameter: 0.467 (poor sync)", font_size=28),
            Text("Energy: 4.60 pJ", font_size=28)
        ).arrange(DOWN).next_to(mc_title, DOWN, buff=1)
        
        for result in results:
            self.play(Write(result), run_time=0.5)
            self.wait(0.3)
        
        self.wait(2)
        self.play(FadeOut(mc_title), FadeOut(results))
        
        # Scaling results
        scale_title = Text("Scaling Analysis (N=2 to 16)", font_size=32).next_to(title, DOWN, buff=1)
        self.play(Write(scale_title))
        
        scale_results = VGroup(
            Text("N=2-4: Some locking achieved", font_size=28, color=YELLOW),
            Text("N=5-16: NO locking achieved", font_size=28, color=RED),
            Text("Order parameter degrades with N", font_size=28),
            Text("Energy scales linearly with N", font_size=28)
        ).arrange(DOWN).next_to(scale_title, DOWN, buff=1)
        
        for result in scale_results:
            self.play(Write(result), run_time=0.5)
            self.wait(0.3)
        
        self.wait(2)
        
        # Key conclusion
        conclusion = Text("Conclusion: O(1) summation claim is EMPIRICALLY FALSE", 
                        font_size=36, color=RED).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)
        
        self.play(FadeOut(title), FadeOut(scale_title), FadeOut(scale_results), FadeOut(conclusion))
    
    def part6_corrected_framework(self):
        """Corrected research framework."""
        title = Text("Corrected Framework: Oscillator Reservoir Computing", 
                    font_size=48, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Reframing
        reframe = VGroup(
            Text("Original (WRONG):", font_size=32, color=RED),
            Text("Direct feedforward neural acceleration", font_size=24),
            Text("General-purpose GPU replacement", font_size=24),
            Text("O(1) operations", font_size=24),
            Text("", font_size=16),
            Text("Correct (RIGHT):", font_size=32, color=GREEN),
            Text("Fixed dynamical system (reservoir)", font_size=24),
            Text("Trainable readout only", font_size=24),
            Text("Specialized for temporal pattern recognition", font_size=24),
            Text("Operations scale with N", font_size=24)
        ).arrange(DOWN).next_to(title, DOWN, buff=1)
        
        for line in reframe:
            self.play(Write(line), run_time=0.3)
            self.wait(0.1)
        
        self.wait(2)
        self.play(FadeOut(reframe))
        
        # Mathematical formulation
        math_title = Text("Kuramoto Reservoir Model", font_size=36, color=GREEN).next_to(title, DOWN, buff=1)
        self.play(Write(math_title))
        
        kuramoto = Text("d(phi_i)/dt = omega_i + sum(K_ij * sin(phi_j - phi_i)) + xi_i(t)", 
                      font_size=24).next_to(math_title, DOWN, buff=1)
        self.play(Write(kuramoto))
        self.wait(1)
        
        readout = Text("y(t) = W_out * R(x(t))", font_size=24).next_to(kuramoto, DOWN, buff=1)
        self.play(Write(readout))
        self.wait(1)
        
        explanation = Text("R: Readout function (order parameters, phase differences)", 
                         font_size=24).next_to(readout, DOWN, buff=0.5)
        self.play(Write(explanation))
        self.wait(2)
        
        self.play(FadeOut(title), FadeOut(math_title), FadeOut(kuramoto), 
                 FadeOut(readout), FadeOut(explanation))
        
        # 7 Theorems
        theorems_title = Text("7 Rigorous Theorems with Proofs", font_size=36, color=GREEN).to_edge(UP)
        theorems_title.shift(DOWN)
        self.play(Write(theorems_title))
        
        theorem_list = VGroup(
            Text("1. Memory Capacity Bound: MC <= N*K/(K+gamma)", font_size=24),
            Text("2. Universal Approximation: Can approximate any continuous function", font_size=24),
            Text("3. Channel Capacity Bound: C <= N*integral(log(1+SNR)df)", font_size=24),
            Text("4. Energy-Efficiency Bound: E_op >= N*P_osc/f_osc", font_size=24),
            Text("5. Synchronization Time: tau_sync ~ 1/lambda_2(L)", font_size=24),
            Text("6. Memory vs Topology: MC ~ rho(A)*K/(K+gamma)", font_size=24),
            Text("7. Noise Robustness: SNR = K^2*sigma^2_input/(S_xi*Tr[L^-1])", font_size=24)
        ).arrange(DOWN).next_to(theorems_title, DOWN, buff=1)
        
        for theorem in theorem_list:
            self.play(Write(theorem), run_time=0.3)
            self.wait(0.1)
        
        self.wait(2)
        self.play(FadeOut(theorems_title), FadeOut(theorem_list))
    
    def part7_true_use_case(self):
        """True use case analysis."""
        title = Text("True Use Case: Temporal Pattern Recognition at Edge", 
                    font_size=48, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Where ORC is advantageous
        advantage_title = Text("Where ORC is Advantageous:", font_size=32, color=GREEN).next_to(title, DOWN, buff=1)
        self.play(Write(advantage_title))
        
        advantages = VGroup(
            Text("1. Always-On Ultra-Low-Power Sensing", font_size=28),
            Text("Static power acceptable", font_size=22),
            Text("Zero latency advantage", font_size=22),
            Text("", font_size=16),
            Text("2. Temporal Pattern Recognition", font_size=28),
            Text("6x lower energy per sample", font_size=22),
            Text("Demonstrated in Torrejon 2017", font_size=22),
            Text("", font_size=16),
            Text("3. Edge AI with Strict Power Budgets", font_size=28),
            Text("30% lower total power", font_size=22),
            Text("No training required", font_size=22)
        ).arrange(DOWN).next_to(advantage_title, DOWN, buff=1)
        
        for advantage in advantages:
            self.play(Write(advantage), run_time=0.3)
            self.wait(0.1)
        
        self.wait(2)
        self.play(FadeOut(advantage_title), FadeOut(advantages))
        
        # Optimal use case
        optimal_title = Text("Optimal Use Case: Keyword Spotting", font_size=36, color=YELLOW).next_to(title, DOWN, buff=1)
        self.play(Write(optimal_title))
        
        optimal_specs = VGroup(
            Text("Requirements:", font_size=28),
            Text("Always-on listening", font_size=22),
            Text("Ultra-low power (<1 mW)", font_size=22),
            Text("Low latency (<100 ms)", font_size=22),
            Text("Moderate accuracy (>90%)", font_size=22),
            Text("", font_size=16),
            Text("Quantitative Advantage:", font_size=28),
            Text("Energy per detection: 25 uJ vs 80 uJ (3.2x)", font_size=22, color=GREEN),
            Text("Latency: 50 ms vs 100 ms (2x)", font_size=22, color=GREEN),
            Text("Battery life: 200h vs 125h (1.6x)", font_size=22, color=GREEN)
        ).arrange(DOWN).next_to(optimal_title, DOWN, buff=1)
        
        for spec in optimal_specs:
            self.play(Write(spec), run_time=0.3)
            self.wait(0.1)
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(optimal_title), FadeOut(optimal_specs))
        
        # Where ORC is NOT advantageous
        not_title = Text("Where ORC is NOT Advantageous:", font_size=32, color=RED).to_edge(UP)
        self.play(Write(not_title))
        
        not_advantages = VGroup(
            Text("1. High-Precision Inference (10-100x disadvantage)", font_size=28),
            Text("2. High-Throughput Computing (10^6x disadvantage)", font_size=28),
            Text("3. Training/Learning (10-100x disadvantage)", font_size=28),
            Text("4. General-Purpose GPU Replacement (fundamentally wrong paradigm)", font_size=28)
        ).arrange(DOWN).next_to(not_title, DOWN, buff=1)
        
        for not_adv in not_advantages:
            self.play(Write(not_adv), run_time=0.5)
            self.wait(0.2)
        
        self.wait(2)
        self.play(FadeOut(not_title), FadeOut(not_advantages))
    
    def part8_research_roadmap(self):
        """Research roadmap."""
        title = Text("Research Roadmap", font_size=48, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Minimum Viable Experiment
        mve_title = Text("Minimum Viable Experiment: Keyword Spotting", font_size=32, color=GREEN).next_to(title, DOWN, buff=1)
        self.play(Write(mve_title))
        
        mve_specs = VGroup(
            Text("Architecture:", font_size=24),
            Text("Ring oscillators, N=16", font_size=20),
            Text("Small-world topology", font_size=20),
            Text("Phase detector + 8-bit ADC", font_size=20),
            Text("", font_size=16),
            Text("Implementation: FPGA (digital emulation)", font_size=24),
            Text("", font_size=16),
            Text("Success Criteria:", font_size=24),
            Text("Minimum: >85% accuracy, <2 mW, <200 ms", font_size=20),
            Text("Target: >90% accuracy, <1 mW, <100 ms", font_size=20),
            Text("Exceptional: >92% accuracy, <500 uW, <50 ms", font_size=20)
        ).arrange(DOWN).next_to(mve_title, DOWN, buff=1)
        
        for spec in mve_specs:
            self.play(Write(spec), run_time=0.2)
            self.wait(0.1)
        
        self.wait(2)
        self.play(FadeOut(mve_title), FadeOut(mve_specs))
        
        # Go/No-Go Criteria
        gogo_title = Text("Go/No-Go Criteria", font_size=32, color=YELLOW).next_to(title, DOWN, buff=1)
        self.play(Write(gogo_title))
        
        gogo_criteria = VGroup(
            Text("Technical Go (must meet ALL):", font_size=24),
            Text("Accuracy >85% on keyword spotting", font_size=20),
            Text("Power <2 mW", font_size=20),
            Text("Phase noise manageable (<10% degradation)", font_size=20),
            Text("Lock rate >80% across process variation", font_size=20),
            Text("", font_size=16),
            Text("Economic Go:", font_size=24),
            Text("Energy advantage >2x vs digital", font_size=20),
            Text("Cost advantage >1.5x vs digital", font_size=20),
            Text("Performance within 5% of digital", font_size=20)
        ).arrange(DOWN).next_to(gogo_title, DOWN, buff=1)
        
        for criterion in gogo_criteria:
            self.play(Write(criterion), run_time=0.2)
            self.wait(0.1)
        
        self.wait(2)
        self.play(FadeOut(gogo_title), FadeOut(gogo_criteria))
        
        # Timeline
        timeline_title = Text("Timeline: 2 Years, $300k Budget", font_size=32, color=GREEN).next_to(title, DOWN, buff=1)
        self.play(Write(timeline_title))
        
        timeline = VGroup(
            Text("Year 1: Proof of Concept", font_size=24),
            Text("Q1: Software simulation + falsification experiment", font_size=20),
            Text("Q2: FPGA implementation", font_size=20),
            Text("Q3: Training + optimization", font_size=20),
            Text("Q4: Paper submission", font_size=20),
            Text("", font_size=16),
            Text("Year 2 (conditional on Year 1 success):", font_size=24),
            Text("Silicon implementation via MPW", font_size=20),
            Text("Productization assessment", font_size=20)
        ).arrange(DOWN).next_to(timeline_title, DOWN, buff=1)
        
        for item in timeline:
            self.play(Write(item), run_time=0.2)
            self.wait(0.1)
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(timeline_title), FadeOut(timeline))
    
    def part9_conclusion(self):
        """Conclusion."""
        title = Text("Conclusion", font_size=72, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Summary
        summary_title = Text("Summary of Findings", font_size=36, color=GREEN).next_to(title, DOWN, buff=1)
        self.play(Write(summary_title))
        
        summary = VGroup(
            Text("What Was Wrong:", font_size=28, color=RED),
            Text("O(1) summation: FALSE (empirically validated)", font_size=22),
            Text("10^12 endurance: FALSE (literature: 10^4-10^9)", font_size=22),
            Text("Universal Adler: FALSE (fails for ring oscillators)", font_size=22),
            Text("GPU replacement: FALSE (fundamentally different paradigm)", font_size=22),
            Text("", font_size=16),
            Text("What Is Correct:", font_size=28, color=GREEN),
            Text("Oscillator computing IS reservoir computing", font_size=22),
            Text("Lock time scales as O(N^a), a in [0.5, 1.0]", font_size=22),
            Text("Phase noise limits precision to 4-6 bits", font_size=22),
            Text("Most plausible: temporal pattern recognition at edge", font_size=22)
        ).arrange(DOWN).next_to(summary_title, DOWN, buff=1)
        
        for item in summary:
            self.play(Write(item), run_time=0.3)
            self.wait(0.1)
        
        self.wait(2)
        self.play(FadeOut(summary_title), FadeOut(summary))
        
        # Critical insight
        insight = Text("Critical Insight", font_size=48, color=YELLOW).to_edge(UP)
        insight.shift(DOWN)
        self.play(Write(insight))
        
        insight_text = VGroup(
            Text("The original ONC vision was fundamentally flawed", font_size=28),
            Text("because it claimed general-purpose superiority", font_size=28),
            Text("without acknowledging physical constraints.", font_size=28),
            Text("", font_size=16),
            Text("The corrected approach focuses on a specific,", font_size=28),
            Text("realistic use case where oscillator systems", font_size=28),
            Text("have genuine advantages:", font_size=28),
            Text("", font_size=16),
            Text("Temporal pattern recognition at the edge", font_size=32, color=GREEN),
            Text("with strict power budgets.", font_size=32, color=GREEN)
        ).arrange(DOWN).next_to(insight, DOWN, buff=1)
        
        for line in insight_text:
            self.play(Write(line), run_time=0.5)
            self.wait(0.2)
        
        self.wait(2)
        self.play(FadeOut(insight), FadeOut(insight_text))
        
        # Final message
        final = Text("This is the only path to scientifically defensible", font_size=36).to_edge(UP)
        final.shift(DOWN * 2)
        final2 = Text("and potentially impactful research.", font_size=36).next_to(final, DOWN)
        
        self.play(Write(final))
        self.play(Write(final2))
        self.wait(3)
        
        self.play(FadeOut(title), FadeOut(final), FadeOut(final2))


if __name__ == "__main__":
    pass
