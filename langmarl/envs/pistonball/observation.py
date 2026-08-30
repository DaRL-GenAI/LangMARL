"""Observation and action formatting for the Pistonball environment.

Extracted from the original ``src/pistonball/episode_generator.py`` so that the
environment-specific parts live next to the environment, and the training loop
itself stays in :mod:`langmarl.trainer`.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

# Imported at module scope on purpose: when the pistonball extra is missing
# the ImportError reaches the environment registry, which skips registration
# and turns it into an actionable message at make_env() time.
from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1


def parse_action_from_response(response, action_mode: str = "discrete"):
    """
    Parse action from LLM response based on action mode.

    Args:
        response: OpenAI API response object
        action_mode: Action space mode - "discrete" (0-2) or "continuous" (-1.0 to 1.0)

    Returns:
        Parsed action value:
        - discrete: int in [0, 1, 2] (0-retract down, 1-stay, 2-push up)
        - continuous: float in [-1.0, 1.0]
    """
    # Extract text content from response
    try:
        # Handle OpenAI response format
        if hasattr(response, 'output'):
            # New responses API format
            text = ""
            for item in response.output:
                if hasattr(item, 'content'):
                    for content in item.content:
                        if hasattr(content, 'text'):
                            text += content.text
        elif hasattr(response, 'choices'):
            # Chat completions format
            text = response.choices[0].message.content
        else:
            text = str(response)
    except Exception:
        text = str(response)

    text = text.strip()

    # Default actions for each mode
    default_actions = {
        "discrete": 1,      # stay
        "continuous": 0.0,  # stay
    }

    if action_mode == "discrete":
        # Parse discrete action (0, 1, or 2)
        # 0 - retract down, 1 - stay, 2 - push up
        # Look for a single digit 0-2 at the start of the response or after "Action:"
        match = re.search(r'(?:^|Action:\s*)([0-2])\b', text, re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1))

        # Try to find any standalone digit 0-2
        match = re.search(r'\b([0-2])\b', text)
        if match:
            return int(match.group(1))

        # Try to parse action names
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['retract', 'down', 'retract_down']):
            return 0
        elif any(keyword in text_lower for keyword in ['stay', 'hold', 'wait']):
            return 1
        elif any(keyword in text_lower for keyword in ['push_up', 'push up', 'push']):
            return 2

        return default_actions["discrete"]

    elif action_mode == "continuous":
        # Parse continuous action value in [-1.0, 1.0]
        # Look for a float value at the start or after "Action:"
        match = re.search(r'(?:^|Action:\s*)([+-]?\d*\.?\d+)', text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = float(match.group(1))
                return max(-1.0, min(1.0, value))  # Clamp to [-1.0, 1.0]
            except ValueError:
                pass

        # Try to find any float in the response
        match = re.search(r'([+-]?\d*\.?\d+)', text)
        if match:
            try:
                value = float(match.group(1))
                return max(-1.0, min(1.0, value))
            except ValueError:
                pass

        # Try to parse descriptive text
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['maximum push', 'full push', 'push up fast']):
            return 1.0
        elif any(keyword in text_lower for keyword in ['moderate push', 'medium push']):
            return 0.5
        elif any(keyword in text_lower for keyword in ['slight push', 'gentle push', 'light push']):
            return 0.3
        elif any(keyword in text_lower for keyword in ['stay', 'hold', 'no movement']):
            return 0.0
        elif any(keyword in text_lower for keyword in ['slight retract', 'gentle retract', 'light retract']):
            return -0.3
        elif any(keyword in text_lower for keyword in ['moderate retract', 'medium retract']):
            return -0.5
        elif any(keyword in text_lower for keyword in ['maximum retract', 'full retract', 'retract fast']):
            return -1.0

        return default_actions["continuous"]

    return default_actions.get(action_mode, 1)



def reset_pistons_to_lowest(env):
    """
    Reset all pistons to their lowest position (maximum y value).

    This function directly manipulates the pymunk physics bodies to set
    all pistons to the lowest position after environment reset.

    Args:
        env: The Pistonball environment (can be wrapped)
    """
    # Get the unwrapped raw environment
    raw = env.unwrapped

    # Calculate the maximum y position (lowest point for pistons)
    # This matches the calculation in pistonball.py
    maximum_piston_y = raw.screen_height - raw.wall_width - (raw.piston_height - raw.piston_head_height)

    # Set each piston to the lowest position
    for piston in raw.pistonList:
        # Keep x position, set y to maximum (lowest point)
        piston.position = (piston.position.x, maximum_piston_y)
        # Reset velocity to zero
        piston.velocity = (0, 0)



def make_env(num_pistons: int, max_cycles: int, frame_size: Tuple[int, int] = (64, 64),
             stack_size: int = 4, continuous: bool = False, render_mode: str = "rgb_array",
             random_drop: bool = True):
    """
    Create a Pistonball environment with preprocessing

    Args:
        num_pistons: Number of pistons in the environment
        max_cycles: Maximum number of cycles per episode
        frame_size: Size of the observation frames
        stack_size: Number of frames to stack
        continuous: Whether to use continuous action space
        render_mode: Render mode for the environment
        random_drop: If True, ball spawns at random x position; if False, ball spawns at x=800

    Returns:
        Preprocessed Pistonball environment
    """
    env = pistonball_v6.parallel_env(
        n_pistons=num_pistons,
        render_mode=render_mode,
        continuous=continuous,
        max_cycles=max_cycles,
        random_drop=random_drop,
    )
    env = color_reduction_v0(env)
    env = resize_v1(env, frame_size[0], frame_size[1])
    env = frame_stack_v1(env, stack_size=stack_size)
    return env



class PistonballObservationFormatter:
    """Format Pistonball observations for LLM understanding using environment state"""

    # Discrete action space (3 actions)
    DISCRETE_ACTION_NAMES = {
        2: "push_up",      # Move piston up to push ball left
        1: "stay",         # Hold current position
        0: "retract_down"  # Move piston down/retract
    }

    DISCRETE_ACTION_DESCRIPTIONS = {
        2: ("push_up", "Move piston UP to push the ball toward the LEFT (goal direction). "
            "Use when ball is above you or approaching."),
        1: ("stay", "HOLD current position. Use to maintain contact with ball or wait."),
        0: ("retract_down", "Move piston DOWN/retract. Use to let ball pass or reset position.")
    }

    # Continuous action space description
    CONTINUOUS_ACTION_RANGE = {
        "min": -1.0,  # Maximum downward (retract)
        "max": 1.0,   # Maximum upward (push)
        "description": "Continuous value from -1.0 (full retract) to 1.0 (full push up)"
    }

    # Legacy mapping for backward compatibility
    ACTION_NAMES = DISCRETE_ACTION_NAMES

    @staticmethod
    def get_env_state(env) -> Dict:
        """
        Extract state information directly from the Pistonball environment.

        Args:
            env: The Pistonball environment (can be wrapped)

        Returns:
            Dictionary containing:
            - ball_position: (x, y) coordinates
            - ball_velocity: (vx, vy) velocity
            - ball_radius: ball radius
            - pistons: list of piston states (position, velocity)
            - screen_width, screen_height: dimensions
            - piston_width, piston_height: piston dimensions
        """
        # Unwrap to get the raw environment
        raw = env.unwrapped

        # Get ball state
        ball = raw.ball
        ball_pos = (float(ball.position.x), float(ball.position.y))
        ball_vel = (float(ball.velocity.x), float(ball.velocity.y))

        # Get piston states
        pistons = []
        for i, piston in enumerate(raw.pistonList):
            pistons.append({
                "index": i,
                "position": (float(piston.position.x), float(piston.position.y)),
                "velocity": (float(piston.velocity.x), float(piston.velocity.y)),
            })

        return {
            "ball_position": ball_pos,
            "ball_velocity": ball_vel,
            "ball_radius": raw.ball_radius,
            "pistons": pistons,
            "screen_width": raw.screen_width,
            "screen_height": raw.screen_height,
            "piston_width": raw.piston_width,
            "piston_height": raw.piston_height,
            "n_pistons": raw.n_pistons,
        }

    @staticmethod
    def format_local_observation(env, agent_idx: int) -> str:
        """
        Format a single agent's local observation as text using environment state.

        Args:
            env: The Pistonball environment
            agent_idx: Index of the agent (0 = leftmost piston)

        Returns:
            Text description of this agent's local observation
        """
        state = PistonballObservationFormatter.get_env_state(env)

        num_agents = state["n_pistons"]
        ball_x, ball_y = state["ball_position"]
        ball_vx, ball_vy = state["ball_velocity"]
        piston = state["pistons"][agent_idx]
        piston_x, piston_y = piston["position"]
        screen_width = state["screen_width"]
        piston_width = state["piston_width"]

        ball_y = 491.0 - ball_y
        piston_y = 491.0 - piston_y

        # Calculate relative position of ball to this piston
        ball_rel_x = ball_x - piston_x  # positive = ball is to the right

        # Determine piston's global position description
        if agent_idx < num_agents * 0.33:
            piston_region = "left side (near goal)"
        elif agent_idx < num_agents * 0.67:
            piston_region = "center"
        else:
            piston_region = "right side (far from goal)"

        # Calculate distances
        distance_to_ball = abs(ball_rel_x)

        lines = [
            "=== Your Piston Status ===",
            f"Piston index: #{agent_idx} of {num_agents} ({piston_region})",
            f"Your X position: {piston_x:.1f} (screen width: {screen_width})",
            f"Your Y position: {piston_y:.1f}",
            "",
            "=== Ball Relative to You ===",
        ]

        # Describe ball position relative to this piston
        if distance_to_ball < piston_width * 1.5:
            lines.append(f"Ball distance: {distance_to_ball:.1f} pixels - VERY CLOSE!")
            if ball_rel_x < 0:
                lines.append("Ball is slightly to your LEFT")
            elif ball_rel_x > 0:
                lines.append("Ball is slightly to your RIGHT")
            else:
                lines.append("Ball is directly ABOVE you")
        elif distance_to_ball < piston_width * 3:
            lines.append(f"Ball distance: {distance_to_ball:.1f} pixels - NEARBY")
            if ball_rel_x < 0:
                lines.append("Ball is to your LEFT (closer to goal)")
            else:
                lines.append("Ball is to your RIGHT (coming towards you)")
        else:
            lines.append(f"Ball distance: {distance_to_ball:.1f} pixels - FAR")
            if ball_rel_x < 0:
                lines.append("Ball has passed you (already closer to goal)")
            else:
                lines.append("Ball has not reached you yet")

        # Ball movement
        lines.append("")
        lines.append("=== Ball Movement ===")
        if abs(ball_vx) < 5:
            h_movement = "nearly stationary horizontally"
        elif ball_vx < 0:
            h_movement = f"moving LEFT at {abs(ball_vx):.1f} px/step (toward goal!)"
        else:
            h_movement = f"moving RIGHT at {ball_vx:.1f} px/step (away from goal)"
        lines.append(f"Horizontal: {h_movement}")

        if abs(ball_vy) < 5:
            v_movement = "nearly stationary vertically"
        elif ball_vy > 0:
            v_movement = f"moving DOWN at {ball_vy:.1f} px/step (toward pistons)"
        else:
            v_movement = f"moving UP at {abs(ball_vy):.1f} px/step (away from pistons)"
        lines.append(f"Vertical: {v_movement}")

        return "\n".join(lines)

    @staticmethod
    def format_global_state(env) -> str:
        """
        Format the global state of the Pistonball environment for LLM.

        Args:
            env: The Pistonball environment

        Returns:
            Formatted state description
        """
        state = PistonballObservationFormatter.get_env_state(env)

        ball_x, ball_y = state["ball_position"]
        ball_y = 491.0 - ball_y
        ball_vx, ball_vy = state["ball_velocity"]
        screen_width = state["screen_width"]
        num_agents = state["n_pistons"]
        piston_width = state["piston_width"]

        # Calculate ball's global position as percentage
        ball_progress = 1.0 - (ball_x / screen_width)  # 1.0 = at left edge (goal)

        # Determine which pistons are near the ball
        nearby_pistons = []
        for p in state["pistons"]:
            dist = abs(p["position"][0] - ball_x)
            if dist < piston_width * 2:
                nearby_pistons.append(p["index"])

        state_lines = [
            "=== Pistonball Global State ===",
            f"Number of pistons: {num_agents}",
            "Objective: Push the ball to the LEFT edge (x=0)",
            "",
            "=== Ball Status ===",
            f"Position: x={ball_x:.1f}, y={ball_y:.1f}",
            f"Progress toward goal: {ball_progress*100:.1f}% (100% = goal reached)",
            f"Velocity: vx={ball_vx:.1f}, vy={ball_vy:.1f}",
        ]

        # Ball location description
        if ball_progress > 0.8:
            state_lines.append("Location: Very close to GOAL! Keep pushing!")
        elif ball_progress > 0.5:
            state_lines.append("Location: Past halfway, good progress")
        elif ball_progress > 0.2:
            state_lines.append("Location: Still in the right half of the field")
        else:
            state_lines.append("Location: Far from goal, need coordinated pushing")

        # Movement summary
        if ball_vx < -10:
            state_lines.append("Movement: Moving LEFT toward goal (good!)")
        elif ball_vx > 10:
            state_lines.append("Movement: Moving RIGHT away from goal (bad!)")
        else:
            state_lines.append("Movement: Nearly stationary horizontally")

        # Nearby pistons
        state_lines.append("")
        state_lines.append("=== Pistons Near Ball ===")
        if nearby_pistons:
            state_lines.append(f"Pistons that can contact ball: {nearby_pistons}")
        else:
            state_lines.append("No pistons currently near the ball")

        # All piston positions summary
        state_lines.append("")
        state_lines.append("=== All Piston Positions ===")
        for p in state["pistons"]:
            px, py = p["position"]
            py = 491.0 - py
            dist_to_ball = ball_x - px
            # Determine NEAR BALL status with left/right direction
            if p["index"] in nearby_pistons:
                if dist_to_ball > 0:
                    status = ">>NEAR BALL (piston is ON LEFT of the ball)<<"  # Ball is to the right of piston
                elif dist_to_ball < 0:
                    status = ">>NEAR BALL (piston is ON RIGHT of the ball)<<"   # Ball is to the left of piston
                else:
                    status = ">>NEAR BALL (piston is at the same x with the ball)<<"  # Ball is directly above piston
            else:
                status = ""
            state_lines.append(f"  Piston {p['index']}: x={px:.0f}, y={py:.0f} (height), ball is {dist_to_ball:+.0f}px away {status}")

        return "\n".join(state_lines)

    @staticmethod
    def format_agent_prompt(env, agent_name: str, policy: str = None,
                            action_mode: str = "discrete") -> str:
        """
        Generate a complete prompt for a specific agent.

        Args:
            env: The Pistonball environment
            agent_name: Name of the agent to generate prompt for
            policy: Optional policy text
            action_mode: "discrete" (3 actions) or "continuous" [-1, 1]

        Returns:
            Complete prompt text for the LLM
        """
        raw = env.unwrapped
        num_agents = raw.n_pistons
        agent_idx = int(agent_name.split("_")[-1])  # Extract index from "piston_X"

        # Get local observation description
        local_obs_text = PistonballObservationFormatter.format_local_observation(env, agent_idx)

        # No global state here on purpose: execution is decentralized, so an
        # actor's prompt carries only its own local observation.
        prompt = f"""
=== GAME RULES ===
You are operating in a **cooperative piston-based control environment**, where you control **piston #{agent_idx}** in a **Pistonball game**.

### Role and Objective
* The environment consists of **{num_agents} pistons**, all of which **share the same reward**
* The collective goal is to **coordinate and push a ball to the LEFT boundary (x = 0)**
* Once the ball reaches the left edge, **all pistons receive the reward**

### Piston Dynamics
* Pistons can move **only vertically**
  * **UP**: pushes the ball to the LEFT (toward the goal)
  * **DOWN**: retracts the piston
* Effective ball movement requires **coordination with neighboring pistons**

### Local Observations
At each timestep, you can observe only **local information**:
1. **Your own vertical height** (in pixels)
2. **The vertical height of your left neighbor** (in pixels)
3. **The vertical height of your right neighbor** (in pixels)
4. **Whether the ball is visible in your local field of view**
5. If the ball is visible:
   * **The horizontal position of the ball relative to you**
   * A **positive value** means the ball is to your right
   * A **negative value** means the ball is to your left

### Action Requirement
* At every timestep, you must choose one discrete action

=== YOUR LOCAL OBSERVATION ===
{local_obs_text}
"""

        if policy:
            prompt += f"""
=== CURRENT POLICY ===
{policy}
"""

        # Add decision section based on action mode
        if action_mode == "discrete":
            prompt += """
=== YOUR DECISION ===
Choose your action (respond with the number only on the first line):
  0 = RETRACT_DOWN
  1 = STAY (hold position)
  2 = PUSH_UP

Action:"""
        elif action_mode == "continuous":
            prompt += """
=== YOUR DECISION ===
Choose your action as a continuous value between -1.0 and 1.0:
  -1.0 = Maximum retract down
   0.0 = Stay in place
  +1.0 = Maximum push up

Respond with a single decimal number (e.g., 0.5, -0.3, 1.0) on the first line.

Action:"""

        return prompt

    @staticmethod
    def format_action(action, action_mode: str = "discrete") -> str:
        """
        Convert action to readable string.

        Args:
            action: Action value (int for discrete, float for continuous)
            action_mode: "discrete" or "continuous"

        Returns:
            Human-readable action description
        """
        if action_mode == "discrete":
            # 0 - retract down, 1 - stay, 2 - push up
            discrete_names = {0: "retract_down", 1: "stay", 2: "push_up"}
            return discrete_names.get(action, f"unknown({action})")
        elif action_mode == "continuous":
            if action > 0.5:
                return f"push_up_strong({action:+.2f})"
            elif action > 0.1:
                return f"push_up_light({action:+.2f})"
            elif action > -0.1:
                return f"stay({action:+.2f})"
            elif action > -0.5:
                return f"retract_light({action:+.2f})"
            else:
                return f"retract_strong({action:+.2f})"
        return f"unknown({action})"

    @staticmethod
    def discrete_to_continuous(action: int) -> float:
        """
        Convert basic discrete action (0,1,2) to continuous value.

        Args:
            action: Discrete action (0=retract_down, 1=stay, 2=push_up)

        Returns:
            Continuous value in [-1.0, 1.0]
        """
        # 0 - retract down -> -1.0, 1 - stay -> 0.0, 2 - push up -> 1.0
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        return mapping.get(action, 0.0)

    @staticmethod
    def continuous_to_discrete(value: float) -> int:
        """
        Convert continuous value to basic discrete action.

        Args:
            value: Continuous value in [-1.0, 1.0]

        Returns:
            Discrete action (0=retract_down, 1=stay, 2=push_up)
        """
        if value > 0.33:
            return 2  # push_up
        elif value < -0.33:
            return 0  # retract_down
        else:
            return 1  # stay

    @staticmethod
    def format_joint_action(actions: Dict, action_mode: str = "discrete") -> str:
        """
        Format joint action of all pistons.

        Args:
            actions: Dictionary of actions keyed by agent name
            action_mode: "discrete" or "continuous"

        Returns:
            Formatted action description
        """
        action_strs = []
        for agent_name, action in sorted(actions.items()):
            action_str = PistonballObservationFormatter.format_action(action, action_mode)
            action_strs.append(f"{agent_name}: {action_str}")
        return ", ".join(action_strs)


