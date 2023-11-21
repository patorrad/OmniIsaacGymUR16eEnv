import xml.etree.ElementTree as ET

file_path = 'omniisaacgymenvs/cfg/bin.xml'

tree = ET.parse(file_path)
root = tree.getroot()

# Now you can iterate over the elements and extract the data
for group_state in root.findall('group_state'):
    state_name = group_state.get('name')
    group_name = group_state.get('group')
    
    print(f"Group State: {state_name}, Group: {group_name}")
    
    for joint in group_state.findall('joint'):
        joint_name = joint.get('name')
        joint_value = float(joint.get('value'))
        
        print(f"    Joint: {joint_name}, Value: {joint_value}")
