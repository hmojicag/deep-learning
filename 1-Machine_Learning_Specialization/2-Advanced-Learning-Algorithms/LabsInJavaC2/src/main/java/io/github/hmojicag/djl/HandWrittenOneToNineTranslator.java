package io.github.hmojicag.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class HandWrittenOneToNineTranslator implements Translator<float[], Integer> {

    @Override
    public Integer processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
        NDArray datum = ndList.get(0);
        float[] y = datum.toFloatArray();
        if (y.length > 0) {
            int maxIndex = 0;
            for (int i = 1; i < y.length; i++) {
                if (y[i] > y[maxIndex]) {
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
        throw new Exception("Output doesn't have any values");
    }

    @Override
    public NDList processInput(TranslatorContext translatorContext, float[] floats) throws Exception {
        NDManager manager = translatorContext.getNDManager();
        NDArray datum = manager.create(floats);
        return new NDList(datum);
    }
}
