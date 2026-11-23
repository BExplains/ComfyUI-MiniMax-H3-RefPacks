import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "BExplains.RefPacks.H3.Migration",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const oldLayouts = {
      BExplainsH3EncodeImages: ["name", "target", "reference_megapixels"],
      BExplainsH3PackLatent: ["kind", "target", "name", "fps"],
      BExplainsH3ImportRefMod: ["filename", "target", "name"]
    };
    if (oldLayouts[nodeData.name]) {
      const configured = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        const result = configured?.apply(this, arguments);
        const layout = oldLayouts[nodeData.name];
        const values = info?.widgets_values;
        if (Array.isArray(values) && ["H3 Image", "H3 Video"].includes(values[layout.indexOf("target")])) {
          for (const widget of this.widgets ?? []) {
            const index = layout.indexOf(widget.name);
            if (index >= 0 && widget.name !== "target") widget.value = values[index];
          }
        }
        return result;
      };
      return;
    }
    if (["BExplainsH3RefPackConditioning", "BExplainsRefPackToH3", "BExplainsInspectRefPacks"].includes(nodeData.name)) {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = created?.apply(this, arguments);
        const widget = this.widgets?.find(w => w.name === "compress_motion");
        if (widget) widget.label = "Compress Motion";
        const rawWidget = this.widgets?.find(w => w.name === "allow_raw_prompt");
        if (rawWidget) rawWidget.label = "Allow Raw Prompt";
        return result;
      };
      return;
    }
  }
});
