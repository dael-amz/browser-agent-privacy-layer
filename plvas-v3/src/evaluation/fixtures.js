import recordsRaw from "../../training/tmp/visual-ui-fixtures/records/validation.jsonl?raw";
import webPiiRecordsRaw from "../../training/tmp/evaluations/plva-visual-agpl-test-v2-webpii-quick100/truth/test.jsonl?raw";

const imageModules = import.meta.glob(
  "../../training/tmp/visual-ui-fixtures/images/validation/*.png",
  { eager: true, query: "?url", import: "default" },
);

const imageByName = new Map(
  Object.entries(imageModules).map(([path, url]) => [path.split("/").at(-1), url]),
);

const webPiiImageModules = import.meta.glob(
  "../../training/tmp/evaluations/plva-visual-agpl-test-v2-webpii-quick100/images/test/*.png",
  { eager: true, query: "?url", import: "default" },
);

export const syntheticValidationRaw = recordsRaw;
export const webPiiQuick100Raw = webPiiRecordsRaw;

export const validationFixtures = recordsRaw
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line))
  .map((record) => {
    const imageName = record.image.split("/").at(-1);
    const imageUrl = imageByName.get(imageName);
    if (!imageUrl) throw new Error(`missing bundled fixture image: ${imageName}`);
    return { ...record, imageUrl };
  });

const webPiiImageByName = new Map(
  Object.entries(webPiiImageModules).map(([path, url]) => [path.split("/").at(-1), url]),
);

export const webPiiQuick100Fixtures = webPiiRecordsRaw
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line))
  .map((record) => {
    const imageName = record.image.split("/").at(-1);
    const imageUrl = webPiiImageByName.get(imageName);
    if (!imageUrl) throw new Error(`missing WebPII diagnostic image: ${imageName}`);
    return { ...record, imageUrl };
  });
