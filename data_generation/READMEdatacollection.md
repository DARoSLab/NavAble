# Data Collection with Drones in AirSim

Requirements:
- **AirSim** is downloaded and integrated into your Unreal Engine environment.
- **Visual Studio 2022** is installed.
- Have a valid `settings.json` configuration file.

### Setup

- Open your Unreal Engine project and load `orbit_bb.py`.
- In your Unreal Engine Editor, place the Player Start pawn in the desired position and rotation.
- In World Setting, set the gamemode to AirSim.

### Configuring Drone Camera

- In Content Browser, open the drone pawn inside Content/Airsim/BluePrints/BP_FlyingPawn.
- Adjust the cameras angles if needed.
- Disable shadow cast on all props and the drone pawn itself.

### Modify the Script

- Edit `orbit_bb.py` to only detect target objects.

### Run the Script and Monitor

- When prompted, input the environment name and run number.
- In Unreal Engine Editor, press Alt+P (or click Play) to start the simulation.
- The script will create a folder in the current directory and save all captured images inside that folder.
- A total of 30 images will be collected from Frame 0 -> Frame 29.