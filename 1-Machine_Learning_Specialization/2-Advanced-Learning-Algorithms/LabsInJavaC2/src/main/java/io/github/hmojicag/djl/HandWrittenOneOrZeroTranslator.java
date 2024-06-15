package io.github.hmojicag.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class HandWrittenOneOrZeroTranslator implements Translator<float[], float[]> {
    @Override
    public float[] processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
        NDArray datum = ndList.get(0);
        return datum.toFloatArray();
    }

    @Override
    public NDList processInput(TranslatorContext translatorContext, float[] floats) throws Exception {
        NDManager manager = translatorContext.getNDManager();
        NDArray datum = manager.create(floats);
        return new NDList(datum);
    }
}
