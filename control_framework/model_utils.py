# python e:\S\PhD\proj\controller_framework\demo\mujoco\dummy_robot.py
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np

BLUE_RGBA = "0.2 0.4 1 0.3"

def _find_worldbody(root):
    return root.find("worldbody")

def _iter_bodies(body):
    yield body
    for child in body.findall("body"):
        yield from _iter_bodies(child)

def _copy_body_recursive(src_body, suffix):
    dst = ET.Element("body", dict(src_body.attrib))
    if "name" in dst.attrib and dst.attrib["name"]:
        dst.attrib["name"] = dst.attrib["name"] + suffix
    for child in list(src_body):
        if child.tag == "inertial":
            continue
        if child.tag == "joint":
            continue
        if child.tag == "geom":
            new_attrib = dict(child.attrib)
            if "name" in new_attrib and new_attrib["name"]:
                new_attrib["name"] = new_attrib["name"] + suffix
            newg = ET.Element("geom", new_attrib)
            newg.set("contype", "0")
            newg.set("conaffinity", "0")
            if child.attrib.get("class") == "visual":
                newg.set("rgba", BLUE_RGBA)
            dst.append(newg)
            continue
        if child.tag == "site":
            new_attrib = dict(child.attrib)
            if "name" in new_attrib and new_attrib["name"]:
                new_attrib["name"] = new_attrib["name"] + suffix
            dst.append(ET.Element("site", new_attrib))
            continue
        if child.tag == "body":
            dst.append(_copy_body_recursive(child, suffix))
            continue
        dst.append(ET.Element(child.tag, dict(child.attrib)))
    return dst

def build_dummy_robot(input_xml, output_xml, root_body_name=None, suffix="_dummy"):
    tree = ET.parse(input_xml)
    root = tree.getroot()
    world = _find_worldbody(root)
    if world is None:
        raise RuntimeError("no worldbody")
    src_root = None
    if root_body_name:
        for b in world.findall("body"):
            if b.get("name") == root_body_name:
                src_root = b
                break
    if src_root is None:
        bodies = world.findall("body")
        if not bodies:
            raise RuntimeError("no robot body")
        src_root = bodies[0]
    dummy_root = _copy_body_recursive(src_root, suffix)
    world.append(dummy_root)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)

def align_dummy_state(model, data, suffix="_dummy"):
    nb = model.nbody
    for bid in range(nb):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        dummy_name = name + suffix
        dummy_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, dummy_name)
        if dummy_id < 0:
            continue
        mocapid = model.body_mocapid[dummy_id]
        if mocapid < 0:
            continue
        data.mocap_pos[mocapid] = data.xpos[bid]
        data.mocap_quat[mocapid] = data.xquat[bid]

if __name__ == "__main__":
    inp = "e:\\S\\PhD\\proj\\controller_framework\\demo\\mujoco\\skyvortex.xml"
    outp = "e:\\S\\PhD\\proj\\controller_framework\\demo\\mujoco\\skyvortex_with_dummy.xml"
    build_dummy_robot(inp, outp, root_body_name="base_link")