// Paste in Firefox console on https://www.minecraft-schematics.com
// Make sure Firefox download directory is set to /home/nixos/schematics_scraped

function dl(id) {
  const a = document.createElement('a');
  a.href = `/schematic/${id}/download/action/?type=schematic`;
  a.download = `${id}.schematic`;
  a.click();
}

const start = 1, end = 30000;
let count = 0;
for (let id = start; id <= end; id++) {
  dl(id);
  await new Promise(r => setTimeout(r, 300));
  if (id % 100 === 0) console.log(`${id}/${end} (${count} fired)`);
  count++;
}
