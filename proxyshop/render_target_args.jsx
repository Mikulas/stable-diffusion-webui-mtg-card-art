#include "scripts/render.jsx";
#include "settings.jsx";

if (arguments.length > 0) {
    file = [new File( arguments[0] )];
} else {
    file = app.openDialog();
}

// Render the selected image
if (file[0]) {

    // Are templates an array
    try {
        if (specified_template === null) render(file[0], specified_template);
        else if ((specified_template[1] !== undefined) && specified_template[1] !== null) {

            // Run through each template
            for (var z = 0; z < specified_template.length; z++) {
                render(file[0], specified_template[z]);
            }

        } else render(file[0], specified_template);
    } catch (error) {
    }

}

if (arguments.length > 1) {
    var jpgOptions = new JPEGSaveOptions();
    jpgOptions.quality = 12;
    jpgOptions.embedColorProfile = true;
    jpgOptions.formatOptions = FormatOptions.PROGRESSIVE;
    if (jpgOptions.formatOptions == FormatOptions.PROGRESSIVE) {
        jpgOptions.scans = 5
    }
    jpgOptions.matte = MatteType.NONE;

    activeDocument.saveAs(new File(arguments[1]), jpgOptions, true, Extension.LOWERCASE);

    activeDocument.close(SaveOptions.DONOTSAVECHANGES);
}
