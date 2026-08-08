"""
SenaAIgent API - Flask application with endpoints for ML analytics, image generation,
aesthetic analysis, and agent orchestration.
"""

import os
import time
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context, make_response

from agents import (
    ModelAgent,
    ImageAgent,
    ArtAgent,
    GemmaAgent,
    VideoAgent,
    CoderAgent,
    AutoHealer,
    DebateEngine,
    RAGAgent,
    AutonomousEvolutionEngine,
    SelfOptimizingVisualTrainer,
    EdgeDiffusionOptimizer,
    DistributedClusterManager,
    OrchestratorAgent,
    TaskPriority,
    RecurrencePattern,
    get_spine,
)


def create_app():
    """Application factory for the Flask app."""
    app = Flask(__name__, static_folder='../static', static_url_path='/static')

    # Initialize Master Spine & Agents
    spine = get_spine()
    gemma_agent = GemmaAgent()
    model_agent = ModelAgent(gemma_agent=gemma_agent)
    image_agent = ImageAgent()
    art_agent = ArtAgent(gemma_agent=gemma_agent)
    video_agent = VideoAgent(gemma_agent=gemma_agent)
    coder_agent = CoderAgent(gemma_agent=gemma_agent)
    auto_healer = AutoHealer(gemma_agent=gemma_agent)
    debate_engine = DebateEngine(gemma_agent=gemma_agent)
    rag_agent = RAGAgent(gemma_agent=gemma_agent)
    evolution_engine = AutonomousEvolutionEngine(
        gemma_agent=gemma_agent,
        coder_agent=coder_agent,
        auto_healer=auto_healer,
        debate_engine=debate_engine,
    )
    visual_trainer = SelfOptimizingVisualTrainer(
        gemma_agent=gemma_agent,
        diffusion_engine=image_agent.diffusion_engine,
        video_agent=video_agent,
    )
    edge_optimizer = EdgeDiffusionOptimizer(
        diffusion_engine=image_agent.diffusion_engine,
        gemma_agent=gemma_agent,
    )
    cluster_manager = DistributedClusterManager()
    orchestrator = OrchestratorAgent()

    # Register agents with orchestrator
    orchestrator.register_agent("gemma", "gemma", gemma_agent, ["generate", "analyze_telemetry", "self_optimize_workflow"])
    orchestrator.register_agent("video", "video", video_agent, ["create_clip"])
    orchestrator.register_agent("coder", "coder", coder_agent, ["synthesize_tool", "execute_tool"])
    orchestrator.register_agent("model", "model", model_agent, ["predict", "analyze"])
    orchestrator.register_agent("image", "image", image_agent, ["generate_image"])
    orchestrator.register_agent("art", "art", art_agent, ["analyze_aesthetics", "create_set"])

    @app.route("/", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "SenaAIgent Pre-AI OS (Gemma Edition)",
            "version": "1.2.0",
            "gemma_status": gemma_agent.get_status(),
            "spine_status": spine.get_status(),
            "endpoints": {
                "health": "/",
                "gemma_intelligence": "/api/gemma",
                "gemma_stream": "/api/gemma/stream",
                "video_creator": "/api/video",
                "master_spine": "/api/spine",
                "coder_synthesizer": "/api/coder",
                "debate_engine": "/api/debate",
                "edge_rag": "/api/rag",
                "water_quality": "/api/water",
                "image_generation": "/api/image",
                "art_analysis": "/api/art",
                "orchestrator": "/api/orchestrator",
            },
        })

    @app.route("/api/gemma", methods=["GET", "POST"])
    def gemma_intelligence():
        """
        Gemma Local LLM Intelligence endpoint.

        GET: Returns status and capabilities of local Gemma agent.
        POST: Executes LLM reasoning, telemetry analysis, or meta-orchestrator optimization.

        POST Body (JSON):
            - action: str ("generate", "analyze_telemetry", "self_optimize")
            - prompt: str (for generate)
            - telemetry: dict (for analyze_telemetry)
        """
        if request.method == "GET":
            return jsonify({
                "endpoint": "/api/gemma",
                "description": "Local Gemma LLM Reasoning & Meta-Orchestrator",
                "status": gemma_agent.get_status(),
                "methods": ["GET", "POST"],
                "actions": {
                    "generate": "Generate text or code using local Gemma model",
                    "analyze_telemetry": "Deep LLM analysis on system/environmental data",
                    "self_optimize": "Autonomous queue and task optimization",
                },
                "example_request": {
                    "action": "generate",
                    "prompt": "Synthesize optimal task execution plan for multi-agent pool",
                },
            })

        try:
            data = request.get_json() or {}
            action = data.get("action", "generate")

            if action == "generate":
                prompt = data.get("prompt", "Hello Gemma")
                system = data.get("system")
                response = gemma_agent.generate(prompt, system=system)
                return jsonify({
                    "success": True,
                    "model": gemma_agent.model_name,
                    "prompt": prompt,
                    "response": response,
                })

            elif action == "analyze_telemetry":
                telemetry = data.get("telemetry", {})
                result = gemma_agent.analyze_telemetry(telemetry)
                return jsonify(result)

            elif action == "self_optimize":
                queue_status = orchestrator.get_queue_status()
                load_metrics = orchestrator.get_load_metrics()
                result = gemma_agent.self_optimize_workflow(queue_status, load_metrics)
                return jsonify(result)

            else:
                return jsonify({"success": False, "error": f"Unknown action: {action}"}), 400

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/gemma/stream", methods=["GET", "POST"])
    def gemma_stream():
        """
        Live SSE Text Streaming Endpoint.
        Streams Gemma token responses word-by-word via Server-Sent Events.
        """
        prompt = request.args.get("prompt") or (request.get_json() or {}).get("prompt", "Hello Gemma")

        def generate_sse():
            full_response = gemma_agent.generate(prompt)
            words = full_response.split(" ")
            for word in words:
                yield f"data: {word} \n\n"
                time.sleep(0.04)
            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate_sse()), content_type="text/event-stream")

    @app.route("/api/video", methods=["GET", "POST"])
    def video_creator():
        """
        Quetzal Video Clip Diffusion Engine Endpoint.

        GET: Returns video creator status and available styles.
        POST: Generates an animated video clip based on prompt and parameters.
        """
        if request.method == "GET":
            return jsonify({
                "endpoint": "/api/video",
                "description": "Quetzal Video Clip Diffusion Engine",
                "methods": ["GET", "POST"],
                "styles": ["quetzal_diffusion", "cybernetic", "surreal", "cosmic"],
                "example_request": {
                    "prompt": "Cybernetic Quetzal flying over emerald city",
                    "width": 512,
                    "height": 512,
                    "num_frames": 16,
                    "fps": 8,
                    "style": "quetzal_diffusion",
                },
            })

        try:
            data = request.get_json() or {}
            prompt = data.get("prompt", "Quetzal Diffusion motion keyframe")
            width = int(data.get("width", 512))
            height = int(data.get("height", 512))
            num_frames = int(data.get("num_frames", 16))
            fps = int(data.get("fps", 8))
            style = data.get("style", "quetzal_diffusion")

            result = video_agent.create_clip(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                style=style,
            )
            return jsonify(result)

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/spine", methods=["GET", "POST"])
    def master_spine_control():
        """
        Master Execution Spine control endpoint.
        """
        if request.method == "GET":
            return jsonify(spine.get_status())

        data = request.get_json() or {}
        if "strict_sequential" in data:
            spine.set_strict_sequential(bool(data["strict_sequential"]))
        return jsonify(spine.get_status())

    @app.route("/api/coder", methods=["GET", "POST"])
    def coder_tool_synthesizer():
        """
        Self-Extending Coder Tool Synthesizer and Coding Engine endpoint.
        """
        if request.method == "GET":
            return jsonify({"description": "Self-Extending Coder Tool Synthesizer & Coding Engine"})
        data = request.get_json() or {}
        action = data.get("action", "synthesize")

        if action == "generate":
            spec = data.get("specification", "")
            lang = data.get("language", "python")
            ctx = data.get("context")
            return jsonify(coder_agent.generate_code(spec, language=lang, context=ctx))
        elif action == "analyze":
            code = data.get("code", "")
            return jsonify(coder_agent.analyze_codebase(code))
        elif action == "refactor":
            code = data.get("code", "")
            goal = data.get("goal", "Refactor and optimize code")
            return jsonify(coder_agent.refactor_code(code, goal))
        elif action == "tests":
            code = data.get("code", "")
            return jsonify(coder_agent.generate_tests(code))
        elif action in ["synthesize", "synthesize_tool"]:
            desc = data.get("task_description") or data.get("requirement", "Process dictionary payload")
            res = coder_agent.synthesize_tool(desc, tool_name=data.get("tool_name"))
            return jsonify(res)
        elif action == "execute":
            res = coder_agent.execute_tool(data.get("tool_name", ""), data.get("payload", {}))
            return jsonify(res)
        return jsonify({"error": f"Unknown action: {action}"}), 400

    @app.route("/api/debate", methods=["GET", "POST"])
    def debate_consensus():
        """
        Multi-Agent Debate & Consensus Engine endpoint.
        """
        if request.method == "GET":
            return jsonify({"description": "Multi-Agent Persona Debate Engine"})
        data = request.get_json() or {}
        topic = data.get("topic", "System Architecture & Security Assessment")
        proposal = data.get("proposal", {})
        rounds = int(data.get("rounds", 2))
        res = debate_engine.run_debate(topic, proposal, rounds=rounds)
        return jsonify(res)

    @app.route("/api/rag", methods=["GET", "POST"])
    def edge_rag():
        """
        Local Edge RAG Engine endpoint.
        """
        if request.method == "GET":
            return jsonify({"description": "Local Edge Zero-Cloud RAG Engine"})
        data = request.get_json() or {}
        action = data.get("action", "query")
        if action == "ingest":
            doc_id = data.get("doc_id", "doc_1")
            text = data.get("text", "")
            chunks = rag_agent.ingest_text(doc_id, text)
            return jsonify({"success": True, "doc_id": doc_id, "chunks_created": chunks})
        elif action in ["ingest_url", "web_ingest", "web_data"]:
            url = data.get("url", "")
            if not url:
                return jsonify({"success": False, "error": "url is required"}), 400
            res = rag_agent.ingest_url(url)
            return jsonify(res)
        elif action == "query":
            res = rag_agent.query(data.get("query", "Summarize ingested documents"))
            return jsonify(res)
        return jsonify({"error": "Unknown action"}), 400

    @app.route("/api/water", methods=["GET", "POST"])
    def water_quality():
        """
        Water quality analysis endpoint.

        GET: Returns API documentation and example usage.
        POST: Analyzes water quality based on provided parameters.

        POST Body (JSON):
            - ph: float (optional, default: 7.0)
            - turbidity: float (optional, default: 1.0)
            - temperature: float (optional, default: 20.0)
            - dissolved_oxygen: float (optional, default: 8.0)

        Returns:
            JSON with water quality analysis results.
        """
        if request.method == "GET":
            return jsonify({
                "endpoint": "/api/water",
                "description": "Water quality analysis using ML analytics",
                "methods": ["GET", "POST"],
                "post_parameters": {
                    "ph": {
                        "type": "float",
                        "description": "pH level (0-14)",
                        "default": 7.0,
                    },
                    "turbidity": {
                        "type": "float",
                        "description": "Turbidity in NTU",
                        "default": 1.0,
                    },
                    "temperature": {
                        "type": "float",
                        "description": "Temperature in Celsius",
                        "default": 20.0,
                    },
                    "dissolved_oxygen": {
                        "type": "float",
                        "description": "Dissolved oxygen in mg/L",
                        "default": 8.0,
                    },
                },
                "example_request": {
                    "ph": 7.2,
                    "turbidity": 2.5,
                    "temperature": 22.0,
                    "dissolved_oxygen": 7.5,
                },
            })

        # POST request - analyze water quality
        try:
            data = request.get_json() or {}

            # Validate numeric types
            params = {}
            for key in ["ph", "turbidity", "temperature", "dissolved_oxygen"]:
                if key in data:
                    try:
                        params[key] = float(data[key])
                    except (TypeError, ValueError):
                        return jsonify({
                            "error": f"Invalid value for {key}: must be a number",
                        }), 400

            result = model_agent.predict_water_quality(params)
            return jsonify({
                "success": True,
                "analysis": result,
            })

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
            }), 500

    @app.route("/api/image", methods=["GET", "POST"])
    def image_generation():
        """
        Image generation endpoint.

        GET: Returns API documentation and available styles.
        POST: Generates an image based on the provided prompt.

        POST Body (JSON):
            - prompt: str (required)
            - width: int (optional, default: 512)
            - height: int (optional, default: 512)
            - style: str (optional)

        Returns:
            JSON with image generation results.
        """
        if request.method == "GET":
            return jsonify({
                "endpoint": "/api/image",
                "description": "AI-powered image generation",
                "methods": ["GET", "POST"],
                "available_styles": image_agent.list_styles(),
                "post_parameters": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image",
                        "required": True,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width (64-2048)",
                        "default": 512,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height (64-2048)",
                        "default": 512,
                    },
                    "style": {
                        "type": "string",
                        "description": "Image style",
                        "required": False,
                    },
                },
                "example_request": {
                    "prompt": "A serene mountain landscape at sunset",
                    "width": 1024,
                    "height": 768,
                    "style": "realistic",
                },
            })

        # POST request - generate image
        try:
            data = request.get_json()

            if not data:
                return jsonify({
                    "success": False,
                    "error": "Request body is required",
                }), 400

            prompt = data.get("prompt")
            if not prompt:
                return jsonify({
                    "success": False,
                    "error": "Prompt is required",
                }), 400

            # Validate prompt
            validation = image_agent.validate_prompt(prompt)
            if not validation["valid"]:
                return jsonify({
                    "success": False,
                    "error": validation["error"],
                }), 400

            # Parse optional parameters
            width = data.get("width", 512)
            height = data.get("height", 512)
            style = data.get("style")

            try:
                width = int(width)
                height = int(height)
            except (TypeError, ValueError):
                return jsonify({
                    "success": False,
                    "error": "Width and height must be integers",
                }), 400

            result = image_agent.generate_image(
                prompt=validation["processed_prompt"],
                width=width,
                height=height,
                style=style,
            )

            if result.get("success"):
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
            }), 500

    @app.route("/api/art", methods=["GET", "POST"])
    def art_analysis():
        """
        Art and aesthetic analysis endpoint.

        GET: Returns API documentation and available styles.
        POST: Analyzes aesthetics or manages image sets.

        POST Body (JSON):
            - action: str (required) - "analyze", "create_set", "add_to_set", "export_context"
            - image_url: str (for analyze)
            - name: str (for create_set)
            - description: str (for create_set)
            - set_id: str (for add_to_set, export_context)

        Returns:
            JSON with analysis or operation results.
        """
        if request.method == "GET":
            return jsonify({
                "endpoint": "/api/art",
                "description": "Art and aesthetic analysis with scalable image sets",
                "methods": ["GET", "POST"],
                "available_styles": art_agent.list_available_styles(),
                "actions": {
                    "analyze": {
                        "description": "Analyze image aesthetics",
                        "params": ["image_url", "image_base64"],
                    },
                    "create_set": {
                        "description": "Create a new scalable image set",
                        "params": ["name", "description", "base_style", "target_count"],
                    },
                    "add_to_set": {
                        "description": "Add image to existing set",
                        "params": ["set_id", "image_url"],
                    },
                    "export_context": {
                        "description": "Export context for other agents",
                        "params": ["set_id"],
                    },
                    "style_transfer": {
                        "description": "Apply style transfer to image",
                        "params": ["image_url", "target_style", "intensity"],
                    },
                },
            })

        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "Request body required"}), 400

            action = data.get("action")
            if not action:
                return jsonify({"success": False, "error": "Action is required"}), 400

            if action == "analyze":
                result = art_agent.analyze_aesthetics({
                    "image_url": data.get("image_url"),
                    "image_base64": data.get("image_base64"),
                })
            elif action == "create_set":
                result = art_agent.create_image_set(
                    name=data.get("name", ""),
                    description=data.get("description", ""),
                    base_style=data.get("base_style"),
                    target_count=data.get("target_count", 10),
                    context=data.get("context"),
                )
            elif action == "add_to_set":
                result = art_agent.add_to_image_set(
                    set_id=data.get("set_id", ""),
                    image_data={"image_url": data.get("image_url")},
                )
            elif action == "export_context":
                result = art_agent.export_context(data.get("set_id", ""))
            elif action == "style_transfer":
                result = art_agent.apply_style_transfer(
                    source_image={"image_url": data.get("image_url")},
                    target_style=data.get("target_style", ""),
                    intensity=data.get("intensity", 0.7),
                )
            else:
                return jsonify({"success": False, "error": f"Unknown action: {action}"}), 400

            if result.get("success"):
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/orchestrator", methods=["GET", "POST"])
    def orchestration():
        """
        Agent orchestration endpoint.

        GET: Returns orchestrator status and documentation.
        POST: Create tasks, check status, or manage agent coordination.

        POST Body (JSON):
            - action: str (required) - "create_task", "get_task", "execute", "status", "analyze"
            - task_type: str (for create_task)
            - payload: dict (for create_task)
            - task_id: str (for get_task, execute)

        Returns:
            JSON with orchestration results.
        """
        if request.method == "GET":
            status = orchestrator.get_queue_status()
            agents = orchestrator.get_agent_status()
            return jsonify({
                "endpoint": "/api/orchestrator",
                "description": "ML-powered agent orchestration and task coordination",
                "methods": ["GET", "POST"],
                "queue_status": status,
                "registered_agents": agents.get("agents", []),
                "actions": {
                    "create_task": {
                        "description": "Create a new task",
                        "params": ["task_type", "payload", "priority"],
                    },
                    "get_task": {
                        "description": "Get task details",
                        "params": ["task_id"],
                    },
                    "execute": {
                        "description": "Execute a specific task",
                        "params": ["task_id"],
                    },
                    "process_queue": {
                        "description": "Process pending tasks",
                        "params": ["max_tasks"],
                    },
                    "status": {
                        "description": "Get queue and agent status",
                        "params": [],
                    },
                    "analyze": {
                        "description": "Analyze workload distribution",
                        "params": [],
                    },
                    "handoff": {
                        "description": "Hand off task to specific agent",
                        "params": ["task_id", "target_agent_id", "context"],
                    },
                },
            })

        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "Request body required"}), 400

            action = data.get("action")
            if not action:
                return jsonify({"success": False, "error": "Action is required"}), 400

            if action == "create_task":
                priority_str = data.get("priority", "MEDIUM").upper()
                priority = getattr(TaskPriority, priority_str, TaskPriority.MEDIUM)
                result = orchestrator.create_task(
                    task_type=data.get("task_type", ""),
                    payload=data.get("payload", {}),
                    priority=priority,
                    depends_on=data.get("depends_on"),
                    context=data.get("context"),
                )
            elif action == "get_task":
                result = orchestrator.get_task(data.get("task_id", ""))
            elif action == "execute":
                result = orchestrator.execute_task(data.get("task_id", ""))
            elif action == "process_queue":
                result = orchestrator.process_queue(data.get("max_tasks"))
            elif action == "status":
                queue = orchestrator.get_queue_status()
                agents = orchestrator.get_agent_status()
                result = {
                    "success": True,
                    "queue": queue,
                    "agents": agents,
                }
            elif action == "analyze":
                result = orchestrator.analyze_workload()
            elif action == "handoff":
                result = orchestrator.handoff_task(
                    task_id=data.get("task_id", ""),
                    target_agent_id=data.get("target_agent_id", ""),
                    additional_context=data.get("context"),
                )
            elif action == "cancel":
                result = orchestrator.cancel_task(data.get("task_id", ""))
            # Scheduler actions
            elif action == "start_scheduler":
                result = orchestrator.start_scheduler(
                    interval=data.get("interval", 5.0),
                    max_tasks_per_cycle=data.get("max_tasks"),
                )
            elif action == "stop_scheduler":
                result = orchestrator.stop_scheduler()
            elif action == "scheduler_status":
                result = orchestrator.get_scheduler_status()
            # Scheduled tasks
            elif action == "schedule_task":
                priority_str = data.get("priority", "MEDIUM").upper()
                priority = getattr(TaskPriority, priority_str, TaskPriority.MEDIUM)
                recurrence_str = data.get("recurrence", "ONCE").upper()
                recurrence = getattr(RecurrencePattern, recurrence_str, RecurrencePattern.ONCE)
                result = orchestrator.schedule_task(
                    task_type=data.get("task_type", ""),
                    payload=data.get("payload", {}),
                    run_at=data.get("run_at"),
                    recurrence=recurrence,
                    priority=priority,
                    context=data.get("context"),
                )
            elif action == "get_scheduled_tasks":
                result = orchestrator.get_scheduled_tasks()
            elif action == "cancel_scheduled":
                result = orchestrator.cancel_scheduled_task(data.get("schedule_id", ""))
            # Webhooks
            elif action == "register_webhook":
                result = orchestrator.register_webhook(
                    webhook_id=data.get("webhook_id", ""),
                    url=data.get("url", ""),
                    events=data.get("events"),
                    headers=data.get("headers"),
                )
            elif action == "unregister_webhook":
                result = orchestrator.unregister_webhook(data.get("webhook_id", ""))
            elif action == "get_webhooks":
                result = orchestrator.get_webhooks()
            # Load management
            elif action == "get_load":
                result = orchestrator.get_load_metrics()
            elif action == "auto_adjust":
                result = orchestrator.auto_adjust_load()
            elif action == "dashboard":
                result = orchestrator.get_system_dashboard()
            # Caching
            elif action == "cache_result":
                result = orchestrator.cache_task_result(
                    task_type=data.get("task_type", ""),
                    payload=data.get("payload", {}),
                    result=data.get("result"),
                    ttl=data.get("ttl"),
                )
            elif action == "get_cached":
                result = orchestrator.get_cached_result(
                    task_type=data.get("task_type", ""),
                    payload=data.get("payload", {}),
                )
            elif action == "clear_cache":
                result = orchestrator.clear_cache(data.get("task_type"))
            elif action == "cache_stats":
                result = {"success": True, "cache": orchestrator.get_cache_stats()}
            elif action == "create_with_cache":
                priority_str = data.get("priority", "MEDIUM").upper()
                priority = getattr(TaskPriority, priority_str, TaskPriority.MEDIUM)
                result = orchestrator.create_task_with_cache(
                    task_type=data.get("task_type", ""),
                    payload=data.get("payload", {}),
                    priority=priority,
                    use_cache=data.get("use_cache", True),
                    context=data.get("context"),
                )
            # Routing
            elif action == "routing_stats":
                result = orchestrator.get_routing_stats()
            elif action == "detect_redundant":
                result = orchestrator.detect_redundant_tasks()
            # Agent pools
            elif action == "create_pool":
                # Get base agent from existing registered agent
                base_agent_id = data.get("base_agent_id", "model")
                base_agent = orchestrator._agents.get(base_agent_id, {}).get("instance")
                if not base_agent:
                    result = {"success": False, "error": f"Base agent {base_agent_id} not found"}
                else:
                    result = orchestrator.create_agent_pool(
                        pool_id=data.get("pool_id", ""),
                        agent_type=data.get("agent_type", ""),
                        base_agent=base_agent,
                        pool_size=data.get("pool_size", 3),
                        capabilities=data.get("capabilities"),
                    )
            elif action == "pool_status":
                result = orchestrator.get_pool_status(data.get("pool_id"))
            elif action == "scale_pool":
                result = orchestrator.scale_pool(
                    pool_id=data.get("pool_id", ""),
                    new_size=data.get("new_size", 3),
                )
            elif action == "parallel_capacity":
                result = orchestrator.get_parallel_capacity()
            # Parallel execution
            elif action == "execute_parallel":
                result = orchestrator.execute_parallel(
                    task_ids=data.get("task_ids", []),
                    max_parallel=data.get("max_parallel"),
                )
            elif action == "process_parallel":
                result = orchestrator.process_queue_parallel(data.get("max_parallel"))
            # External trigger
            elif action == "trigger":
                result = orchestrator.trigger_external(
                    trigger_id=data.get("trigger_id", ""),
                    payload=data.get("trigger_payload"),
                )
            else:
                return jsonify({"success": False, "error": f"Unknown action: {action}"}), 400

            if result.get("success"):
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # =========================================================================
    # OPENAI COMPATIBILITY GATEWAY FOR ANTIGRAVITY IDE & EXTERNAL CLIENTS
    # =========================================================================

    @app.route("/v1/models", methods=["GET"])
    def list_openai_models():
        """OpenAI-compatible models list endpoint."""
        status = gemma_agent.get_status()
        return jsonify({
            "object": "list",
            "data": [
                {
                    "id": status.get("model", "gemma2:2b"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "senaai-gemma",
                    "hardware_acceleration": status.get("hardware_acceleration", "Apple Silicon MPS"),
                },
                {
                    "id": "senaai-coder",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "senaai-gemma",
                },
                {
                    "id": "senaai-diffusion",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "pytorch-mps",
                },
            ],
        })

    @app.route("/v1/chat/completions", methods=["POST"])
    def openai_chat_completions():
        """
        OpenAI-compatible chat completions endpoint for Antigravity IDE integration.
        """
        try:
            data = request.get_json() or {}
            messages = data.get("messages", [])
            model = data.get("model", "gemma2:2b")
            temperature = data.get("temperature", 0.7)

            prompt_parts = []
            system_msg = None
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    system_msg = content
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            prompt = "\n".join(prompt_parts) if prompt_parts else "Hello"
            response_text = gemma_agent.generate(prompt, system=system_msg, temperature=temperature)

            return jsonify({
                "id": f"chatcmpl-senaai-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(prompt.split()) + len(response_text.split()),
                },
            })
        except Exception as e:
            return jsonify({"error": f"OpenAI Gateway Error: {str(e)}"}), 500

    # =========================================================================
    # PYTORCH DIFFUSION ENDPOINT
    # =========================================================================

    @app.route("/api/image/diffusion", methods=["POST"])
    def local_diffusion_api():
        """API endpoint for PyTorch MPS Apple Silicon local image diffusion."""
        try:
            data = request.get_json() or {}
            prompt = data.get("prompt", "")
            width = data.get("width", 512)
            height = data.get("height", 512)
            num_steps = data.get("num_inference_steps", 20)
            neg_prompt = data.get("negative_prompt")
            raw_mode = data.get("raw_mode", False)
            model_id = data.get("model_id")
            quality_preset = data.get("quality_preset", False)
            return jsonify(image_agent.diffusion_engine.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                negative_prompt=neg_prompt,
                raw_mode=raw_mode,
                model_id=model_id,
                quality_preset=quality_preset,
            ))
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/evolution", methods=["GET", "POST"])
    def evolution_cycle_api():
        """Autonomous Evolution & Self-Improvement Cycle API."""
        try:
            if request.method == "GET":
                return jsonify({"status": "ready", "history_cycles": len(evolution_engine.evolution_history)})
            data = request.get_json() or {}
            goal = data.get("goal", "Optimize system performance and repair code bottlenecks")
            return jsonify(evolution_engine.run_evolution_cycle(target_goal=goal))
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/trainer", methods=["POST"])
    def visual_trainer_api():
        """API endpoint for metadata extraction and iterative visual training."""
        try:
            data = request.get_json() or {}
            ref_path = data.get("reference_path", "")
            description = data.get("description", "")
            max_iter = data.get("max_iterations", 3)
            target_score = data.get("target_score", 85.0)

            if not ref_path:
                return jsonify({"success": False, "error": "reference_path is required"}), 400

            return jsonify(visual_trainer.train_image_match_loop(
                reference_path=ref_path,
                user_description=description,
                max_iterations=max_iter,
                target_score=target_score,
            ))
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/edge_optimizer", methods=["GET", "POST"])
    def edge_optimizer_api():
        """API endpoint for autonomous consumer hardware edge diffusion optimization."""
        try:
            if request.method == "POST":
                data = request.get_json() or {}
                target_speed = data.get("target_it_per_sec", 8.0)
                min_quality = data.get("min_quality_score", 80.0)
                return jsonify(edge_optimizer.run_self_optimization_loop(
                    target_it_per_sec=target_speed,
                    min_quality_score=min_quality,
                ))
            return jsonify(edge_optimizer.benchmark_profile())
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/cluster", methods=["GET", "POST"])
    def cluster_api():
        """API endpoint for multi-Mac local network edge cluster management and batch dispatch."""
        try:
            if request.method == "POST":
                data = request.get_json() or {}
                action = data.get("action", "register")
                if action == "register":
                    node_id = data.get("node_id", f"node_{int(time.time())}")
                    url = data.get("url", "")
                    if not url:
                        return jsonify({"success": False, "error": "Node url is required"}), 400
                    return jsonify(cluster_manager.register_node(node_id=node_id, url=url))
                elif action == "batch":
                    prompts = data.get("prompts", [])
                    return jsonify(cluster_manager.distribute_batch_generation(prompts=prompts))
            return jsonify({"success": True, "active_nodes": cluster_manager.get_active_nodes()})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/dashboard", methods=["GET"])
    def dashboard_api():
        """
        Dashboard data endpoint for frontend load gauge and metrics.

        Returns:
            JSON with complete dashboard data including load gauge.
        """
        return jsonify(orchestrator.get_system_dashboard())

    @app.route("/api/load", methods=["GET"])
    def load_metrics():
        """
        Load metrics endpoint for real-time gauge updates.

        Returns:
            JSON with load gauge data and metrics.
        """
    @app.after_request
    def add_no_cache_headers(response):
        """Ensure static files and HTML pages are never cached by the browser."""
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.route("/api/trainer/fine_tune", methods=["POST"])
    def trainer_fine_tune_api():
        """Run full local model fine-tuning & weight optimization loop."""
        try:
            data = request.get_json() or {}
            training_dir = data.get("training_dir", "static/training_data")
            epochs = int(data.get("epochs", 5))
            lr = float(data.get("learning_rate", 1e-4))
            res = visual_trainer.run_full_model_training(training_dir=training_dir, epochs=epochs, learning_rate=lr)
            return jsonify(res)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/trainer/subject", methods=["POST"])
    def trainer_subject_api():
        """Auto-fetch dataset & fine-tune model weights for a specific subject (e.g. roadrunner)."""
        try:
            data = request.get_json() or {}
            subject = data.get("subject", "roadrunner")
            res = visual_trainer.fetch_and_train_subject(subject_name=subject)
            return jsonify(res)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/dashboard")
    def dashboard_page():
        """Serve the dashboard frontend with strict no-cache control."""
        resp = make_response(send_from_directory(app.static_folder, 'dashboard.html'))
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.route("/chat")
    def chat_page():
        """Serve the live chat frontend."""
        return send_from_directory(app.static_folder, 'chat.html')

    @app.errorhandler(404)
    def not_found(e):
        """Handle 404 errors."""
        return jsonify({
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": ["/", "/api/water", "/api/image", "/api/art", "/api/orchestrator", "/api/dashboard", "/api/load", "/dashboard"],
        }), 404

    @app.errorhandler(500)
    def internal_error(e):
        """Handle 500 errors."""
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
        }), 500

    return app


# Create the app instance for Gunicorn
app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
