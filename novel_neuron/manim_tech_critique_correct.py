"""
Manim Video: The Discovery - Technical Critique of Oscillatory Neural Computation
===============================================================================

Visually compelling version with 3D graphics using manimlib API.

Run: manim manim_tech_critique_correct.py TechnicalCritiqueVideo
"""

from manimlib import *
import numpy as np


class TechnicalCritiqueVideo(Scene):
    """Main video scene with 3D graphics."""
    CONFIG = {
        "camera_class": ThreeDCamera,
    }
    
    def construct(self):
        # Part 1: Introduction with 3D
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
        """Introduction with 3D elements."""
        # 3D title
        title = Text("The Discovery")
        title.scale(2)
        title.to_edge(UP)
        
        subtitle = Text("Technical Critique of Oscillatory Neural Computation")
        subtitle.scale(1.2)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title), run_time=2)
        self.play(Write(subtitle), run_time=2)
        self.wait(2)
        
        # 3D rotating sphere representing the problem
        sphere = Sphere(radius=1, color=RED)
        sphere.set_fill(RED, opacity=0.5)
        sphere.move_to(ORIGIN + RIGHT * 3)
        self.play(FadeIn(sphere))
        self.play(Rotate(sphere, angle=PI, axis=UP), run_time=2)
        
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(sphere))
        
        # Problem statement with 3D boxes
        problem = Text("The Problem:")
        problem.scale(1.5)
        problem.set_color(YELLOW)
        problem.to_edge(UP)
        self.play(Write(problem))
        self.wait(1)
        
        # Create 3D boxes for each claim
        claims = [
            "O(1) summation",
            "10^12 endurance",
            "Universal Adler",
            "8x information density",
            "GPU replacement"
        ]
        
        boxes = VGroup()
        for i, claim in enumerate(claims):
            box = Cube(side_length=0.5)
            box.set_fill(BLUE, opacity=0.3)
            box.move_to(LEFT * 4 + RIGHT * i * 2)
            text = Text(claim)
            text.scale(0.5)
            text.move_to(box.get_center())
            text.shift(OUT * 0.3)
            boxes.add(VGroup(box, text))
        
        self.play(LaggedStart(*[Create(box) for box in boxes], lag_ratio=0.2))
        self.play(Rotate(boxes, angle=PI/4, axis=UP), run_time=2)
        
        self.wait(2)
        self.play(FadeOut(problem), FadeOut(boxes))
    
    def part2_original_claims(self):
        """Original claims with 3D comparison."""
        title = Text("Original Claims vs Reality")
        title.scale(1.5)
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 3D comparison - original vs reality
        original_cube = Cube(side_length=1)
        original_cube.set_fill(RED, opacity=0.5)
        original_cube.move_to(LEFT * 3)
        original_label = Text("Original")
        original_label.next_to(original_cube, DOWN)
        
        reality_cube = Cube(side_length=1)
        reality_cube.set_fill(GREEN, opacity=0.5)
        reality_cube.move_to(RIGHT * 3)
        reality_label = Text("Reality")
        reality_label.next_to(reality_cube, DOWN)
        
        self.play(Create(original_cube), Create(reality_cube))
        self.play(Write(original_label), Write(reality_label))
        
        # Animate transformation
        self.play(
            original_cube.scale(0.5),
            reality_cube.scale(1.5),
            run_time=2
        )
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(original_cube), FadeOut(reality_cube), 
                 FadeOut(original_label), FadeOut(reality_label))
        
        # Critical error with 3D warning
        critical = Text("Most Critical Error: O(1) Summation")
        critical.scale(1.5)
        critical.set_color(RED)
        critical.to_edge(UP)
        self.play(Write(critical))
        
        # 3D warning symbol
        warning = Tetrahedron(edge_length=1)
        warning.set_fill(RED, opacity=0.7)
        warning.move_to(ORIGIN + UP * 2)
        self.play(FadeIn(warning))
        self.play(Rotate(warning, angle=2*PI, axis=UP), run_time=2)
        
        explanation = VGroup(
            Text("Original: tau_lock ~ constant"),
            Text("Reality: tau_lock ~ N^a, a in [0.5, 1.0]"),
            Text("Evidence: 0% locking at N>4")
        )
        explanation.arrange(DOWN)
        explanation.next_to(critical, DOWN, buff=1)
        
        for line in explanation:
            self.play(Write(line), run_time=0.5)
            self.wait(0.3)
        
        self.wait(2)
        self.play(FadeOut(critical), FadeOut(warning), FadeOut(explanation))
    
    def part3_literature_survey(self):
        """Literature survey with 3D book representations."""
        title = Text("Literature Survey")
        title.scale(1.5)
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 3D books representing papers
        papers_info = [
            ("Torrejon 2017", GREEN),
            ("Chiba 2024", GREEN),
            ("Hong 2019", GREEN),
            ("Francois 2024", GREEN)
        ]
        
        books = VGroup()
        for i, (name, color) in enumerate(papers_info):
            book = Cube(side_length=0.8)
            book.set_fill(color, opacity=0.4)
            book.move_to(LEFT * 4 + RIGHT * i * 2.5 + UP * 0.5)
            book.rotate(PI/6, axis=RIGHT)
            text = Text(name)
            text.scale(0.4)
            text.move_to(book.get_center() + DOWN * 0.8)
            books.add(VGroup(book, text))
        
        self.play(LaggedStart(*[Create(book) for book in books], lag_ratio=0.3))
        self.play(Rotate(books, angle=PI/6, axis=UP), run_time=2)
        
        self.wait(2)
        
        # Key insight with 3D lightbulb
        insight = Text("Key Insight: RESERVOIR COMPUTING")
        insight.scale(1.2)
        insight.set_color(YELLOW)
        insight.to_edge(DOWN)
        self.play(Write(insight))
        
        lightbulb = Sphere(radius=0.5)
        lightbulb.set_fill(YELLOW, opacity=0.8)
        lightbulb.move_to(ORIGIN + UP * 2)
        self.play(FadeIn(lightbulb))
        self.play(Flash(lightbulb), run_time=1)
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(books), FadeOut(insight), FadeOut(lightbulb))
    
    def part4_mathematical_corrections(self):
        """Mathematical corrections with 3D graphs."""
        title = Text("Mathematical Corrections")
        title.scale(1.5)
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 3D surface plot for memory capacity
        theorem1 = Text("Theorem 1: Memory Capacity Bound")
        theorem1.scale(1.2)
        theorem1.set_color(GREEN)
        theorem1.to_edge(UP)
        theorem1.shift(DOWN)
        self.play(Write(theorem1))
        
        # Create 3D surface
        surface = Surface(
            lambda u, v: np.array([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)]),
            u_range=[0, 2*np.pi],
            v_range=[0, np.pi],
        )
        surface.set_fill(BLUE, opacity=0.5)
        surface.scale(2)
        surface.move_to(ORIGIN + RIGHT * 2)
        
        self.play(Create(surface))
        self.play(Rotate(surface, angle=PI, axis=UP), run_time=3)
        
        self.wait(2)
        self.play(FadeOut(theorem1), FadeOut(surface))
        
        # Energy bound with 3D bar chart
        theorem4 = Text("Theorem 4: Energy-Efficiency Bound")
        theorem4.scale(1.2)
        theorem4.set_color(GREEN)
        theorem4.to_edge(UP)
        theorem4.shift(DOWN)
        self.play(Write(theorem4))
        
        bars = VGroup()
        for i in range(5):
            height = 0.5 + i * 0.3
            bar = Cube(side_length=0.5, depth=0.5, height=height)
            bar.set_fill(RED, opacity=0.6)
            bar.move_to(LEFT * 3 + RIGHT * i * 1.5 + UP * height/2)
            bars.add(bar)
        
        self.play(LaggedStart(*[Create(bar) for bar in bars], lag_ratio=0.2))
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(theorem4), FadeOut(bars))
    
    def part5_simulation_results(self):
        """Simulation results with 3D visualization."""
        title = Text("Simulation Results")
        title.scale(1.5)
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 3D oscillator array visualization
        mc_title = Text("Monte Carlo: N=4, 10 runs")
        mc_title.scale(1.1)
        mc_title.next_to(title, DOWN, buff=1)
        self.play(Write(mc_title))
        
        # Create 3D oscillator spheres
        oscillators = VGroup()
        for i in range(4):
            osc = Sphere(radius=0.3)
            osc.set_fill(BLUE, opacity=0.6)
            osc.move_to(LEFT * 2 + RIGHT * i * 1.5)
            oscillators.add(osc)
        
        self.play(LaggedStart(*[Create(osc) for osc in oscillators], lag_ratio=0.2))
        
        # Animate locking failure
        self.play(
            oscillators[0].set_fill(GREEN, opacity=0.6),
            oscillators[1].set_fill(GREEN, opacity=0.6),
            oscillators[2].set_fill(RED, opacity=0.6),
            oscillators[3].set_fill(GREEN, opacity=0.6),
            run_time=1
        )
        
        results = VGroup(
            Text("Lock rate: 50%"),
            Text("Order parameter: 0.467"),
            Text("Energy: 4.60 pJ")
        )
        results.arrange(DOWN)
        results.next_to(mc_title, DOWN, buff=2)
        results[0].set_color(RED)
        
        for result in results:
            self.play(Write(result), run_time=0.5)
            self.wait(0.3)
        
        self.wait(2)
        self.play(FadeOut(mc_title), FadeOut(oscillators), FadeOut(results))
        
        # Scaling with 3D graph
        scale_title = Text("Scaling: N=2 to 16")
        scale_title.scale(1.1)
        scale_title.next_to(title, DOWN, buff=1)
        self.play(Write(scale_title))
        
        # 3D line graph
        points = VGroup()
        for i in range(8):
            point = Dot3D()
            point.move_to(LEFT * 3 + RIGHT * i * 1 + UP * (0.5 if i < 2 else 0))
            point.set_color(RED)
            points.append(point)
        
        self.play(LaggedStart(*[Create(p) for p in points], lag_ratio=0.2))
        
        scale_results = VGroup(
            Text("N=2-4: Some locking"),
            Text("N=5-16: NO locking")
        )
        scale_results.arrange(DOWN)
        scale_results.next_to(scale_title, DOWN, buff=2)
        scale_results[0].set_color(YELLOW)
        scale_results[1].set_color(RED)
        
        for result in scale_results:
            self.play(Write(result), run_time=0.5)
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(scale_title), FadeOut(points), FadeOut(scale_results))
    
    def part6_corrected_framework(self):
        """Corrected framework with 3D transformation."""
        title = Text("Corrected Framework: Oscillator Reservoir Computing")
        title.scale(1.2)
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 3D transformation from old to new
        old_model = Cube(side_length=1)
        old_model.set_fill(RED, opacity=0.5)
        old_model.move_to(LEFT * 3)
        old_label = Text("Old: Direct Neural")
        old_label.next_to(old_model, DOWN)
        
        new_model = Sphere(radius=0.6)
        new_model.set_fill(GREEN, opacity=0.5)
        new_model.move_to(RIGHT * 3)
        new_label = Text("New: Reservoir")
        new_label.next_to(new_model, DOWN)
        
        self.play(Create(old_model), Create(new_model))
        self.play(Write(old_label), Write(new_label))
        
        # Transform animation
        self.play(
            old_model.scale(0.3),
            old_model.set_fill(GRAY, opacity=0.5),
            new_model.scale(1.5),
            new_model.set_fill(YELLOW, opacity=0.5),
            run_time=2
        )
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(old_model), FadeOut(new_model), 
                 FadeOut(old_label), FadeOut(new_label))
        
        # 7 theorems with 3D pillars
        theorems_title = Text("7 Rigorous Theorems")
        theorems_title.scale(1.2)
        theorems_title.set_color(GREEN)
        theorems_title.to_edge(UP)
        theorems_title.shift(DOWN)
        self.play(Write(theorems_title))
        
        pillars = VGroup()
        for i in range(7):
            height = 1 + i * 0.2
            pillar = Cube(side_length=0.3, depth=0.3, height=height)
            pillar.set_fill(BLUE, opacity=0.6)
            pillar.move_to(LEFT * 3 + RIGHT * i * 1 + UP * height/2)
            pillars.add(pillar)
        
        self.play(LaggedStart(*[Create(p) for p in pillars], lag_ratio=0.1))
        self.play(Rotate(pillars, angle=PI/6, axis=UP), run_time=2)
        
        self.wait(2)
        self.play(FadeOut(theorems_title), FadeOut(pillars))
    
    def part7_true_use_case(self):
        """True use case with 3D application visualization."""
        title = Text("True Use Case: Temporal Pattern Recognition")
        title.scale(1.2)
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 3D keyword spotting visualization
        advantage_title = Text("Optimal: Keyword Spotting")
        advantage_title.scale(1.1)
        advantage_title.set_color(GREEN)
        advantage_title.next_to(title, DOWN, buff=1)
        self.play(Write(advantage_title))
        
        # 3D microphone
        mic = Cylinder(radius=0.3, height=1)
        mic.set_fill(GRAY, opacity=0.7)
        mic.move_to(LEFT * 3)
        mic.rotate(PI/2, axis=RIGHT)
        self.play(Create(mic))
        
        # 3D waveform
        wave_points = VGroup()
        for i in range(20):
            y = np.sin(i * 0.5) * 0.3
            point = Dot3D()
            point.move_to(RIGHT * 2 + LEFT * i * 0.3 + UP * y)
            point.set_color(YELLOW)
            wave_points.append(point)
        
        self.play(LaggedStart(*[Create(p) for p in wave_points], lag_ratio=0.05))
        
        specs = VGroup(
            Text("Energy: 25 uJ vs 80 uJ (3.2x)"),
            Text("Latency: 50 ms vs 100 ms (2x)"),
            Text("Battery: 200h vs 125h (1.6x)")
        )
        specs.arrange(DOWN)
        specs.next_to(advantage_title, DOWN, buff=2)
        for spec in specs:
            spec.set_color(GREEN)
        
        for spec in specs:
            self.play(Write(spec), run_time=0.5)
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(advantage_title), FadeOut(mic), 
                 FadeOut(wave_points), FadeOut(specs))
        
        # Where NOT advantageous with 3D X
        not_title = Text("NOT For:")
        not_title.scale(1.1)
        not_title.set_color(RED)
        not_title.to_edge(UP)
        self.play(Write(not_title))
        
        not_apps = VGroup(
            Text("High-Precision Inference"),
            Text("High-Throughput Computing"),
            Text("Training/Learning")
        )
        not_apps.arrange(DOWN)
        not_apps.next_to(not_title, DOWN, buff=1)
        
        # 3D X marks
        x_marks = VGroup()
        for i in range(3):
            x1 = Line(LEFT * 0.3, RIGHT * 0.3)
            x1.set_stroke(RED, 5)
            x2 = Line(UP * 0.3, DOWN * 0.3)
            x2.set_stroke(RED, 5)
            x = VGroup(x1, x2)
            x.move_to(RIGHT * 4 + UP * (1 - i))
            x_marks.add(x)
        
        self.play(Write(not_apps))
        self.play(LaggedStart(*[Create(x) for x in x_marks], lag_ratio=0.3))
        
        self.wait(2)
        self.play(FadeOut(not_title), FadeOut(not_apps), FadeOut(x_marks))
    
    def part8_research_roadmap(self):
        """Research roadmap with 3D timeline."""
        title = Text("Research Roadmap")
        title.scale(1.5)
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 3D timeline
        timeline_title = Text("Timeline: 2 Years, $300k")
        timeline_title.scale(1.1)
        timeline_title.set_color(GREEN)
        timeline_title.next_to(title, DOWN, buff=1)
        self.play(Write(timeline_title))
        
        # 3D timeline blocks
        blocks = VGroup()
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        for i, quarter in enumerate(quarters):
            block = Cube(side_length=0.8)
            block.set_fill(BLUE, opacity=0.5)
            block.move_to(LEFT * 3 + RIGHT * i * 2)
            label = Text(quarter)
            label.scale(0.5)
            label.move_to(block.get_center() + DOWN * 0.7)
            blocks.add(VGroup(block, label))
        
        self.play(LaggedStart(*[Create(b) for b in blocks], lag_ratio=0.2))
        self.play(Rotate(blocks, angle=PI/8, axis=UP), run_time=2)
        
        tasks = VGroup(
            Text("Software simulation"),
            Text("FPGA implementation"),
            Text("Training + optimization"),
            Text("Paper submission")
        )
        tasks.arrange(DOWN)
        tasks.next_to(timeline_title, DOWN, buff=2)
        
        for task in tasks:
            self.play(Write(task), run_time=0.3)
        
        self.wait(2)
        self.play(FadeOut(title), FadeOut(timeline_title), FadeOut(blocks), FadeOut(tasks))
    
    def part9_conclusion(self):
        """Conclusion with 3D summary."""
        title = Text("Conclusion")
        title.scale(2)
        title.set_color(YELLOW)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 3D summary spheres
        summary_title = Text("Summary")
        summary_title.scale(1.2)
        summary_title.set_color(GREEN)
        summary_title.next_to(title, DOWN, buff=1)
        self.play(Write(summary_title))
        
        # Wrong claims (red spheres)
        wrong_spheres = VGroup()
        for i in range(4):
            sphere = Sphere(radius=0.3)
            sphere.set_fill(RED, opacity=0.6)
            sphere.move_to(LEFT * 3 + RIGHT * i * 1.5 + UP * 1)
            wrong_spheres.add(sphere)
        
        # Correct claims (green spheres)
        correct_spheres = VGroup()
        for i in range(4):
            sphere = Sphere(radius=0.3)
            sphere.set_fill(GREEN, opacity=0.6)
            sphere.move_to(LEFT * 3 + RIGHT * i * 1.5 + DOWN * 1)
            correct_spheres.add(sphere)
        
        self.play(LaggedStart(*[Create(s) for s in wrong_spheres], lag_ratio=0.1))
        self.play(LaggedStart(*[Create(s) for s in correct_spheres], lag_ratio=0.1))
        
        wrong_label = Text("WRONG")
        wrong_label.set_color(RED)
        wrong_label.move_to(UP * 1.5 + LEFT * 3)
        correct_label = Text("CORRECT")
        correct_label.set_color(GREEN)
        correct_label.move_to(DOWN * 1.5 + LEFT * 3)
        self.play(Write(wrong_label), Write(correct_label))
        
        self.wait(2)
        self.play(FadeOut(summary_title), FadeOut(wrong_spheres), FadeOut(correct_spheres), 
                 FadeOut(wrong_label), FadeOut(correct_label))
        
        # Final message with 3D globe
        insight = Text("Temporal Pattern Recognition at Edge")
        insight.scale(1.2)
        insight.set_color(YELLOW)
        insight.to_edge(DOWN)
        self.play(Write(insight))
        
        globe = Sphere(radius=1)
        globe.set_fill(BLUE, opacity=0.3)
        globe.move_to(ORIGIN + UP * 2)
        self.play(FadeIn(globe))
        self.play(Rotate(globe, angle=2*PI, axis=UP), run_time=3)
        
        final = Text("Scientifically Defensible Research")
        final.scale(1.2)
        final.to_edge(UP)
        final.shift(DOWN * 2)
        self.play(Write(final))
        
        self.wait(3)
        self.play(FadeOut(title), FadeOut(globe), FadeOut(insight), FadeOut(final))
