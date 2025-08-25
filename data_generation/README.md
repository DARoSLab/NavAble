# Data Collection with Drones in AirSim

Requirements:
- **AirSim** is downloaded and integrated into your Unreal Engine environment.
- **Visual Studio 2022** is installed.
- Have a valid `settings.json` configuration file.

### Setup

- Open your Unreal Engine project and load `orbit_bb.py`.
- In your Unreal Engine Editor, place the Player Start pawn in the desired position and rotation.
- In World Setting, set the gamemode to AirSim.
<img width="766" height="188" alt="Image" src="https://github.com/user-attachments/assets/3a14d2f8-9c61-423b-a330-812923221c20" />

### Configuring Drone Camera

- In Content Browser, open the drone pawn inside `ContentBrowser/Airsim/Content/BluePrints/BP_FlyingPawn`.
<img width="1163" height="366" alt="Image" src="https://github.com/user-attachments/assets/185906cf-6b1b-4237-ab1f-52a32caa2966" />

- Adjust the cameras angles if needed.
- Disable 'Cast Shadow' on all props and the drone pawn itself.
<img width="357" height="85" alt="Image" src="https://github.com/user-attachments/assets/8aeab6b6-dfb5-40d5-9820-77a6e45a7602" />

### Modify the Script

- Edit `orbit_bb.py` to only detect target objects.

### Run the Script and Monitor

- When prompted, input the environment name and run number.
<img width="405" height="60" alt="Image" src="https://github.com/user-attachments/assets/747700b5-0868-4d2a-97bf-b0e948b68b30" />

- In Unreal Engine Editor, press Alt+P (or click Play) to start the simulation.
- The script will create a folder in the current directory and save all captured images inside that folder.
- A total of 30 images will be collected from Frame 0 -> Frame 29.
