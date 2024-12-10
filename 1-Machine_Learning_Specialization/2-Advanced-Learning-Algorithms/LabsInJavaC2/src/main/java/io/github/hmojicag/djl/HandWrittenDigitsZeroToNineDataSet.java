package io.github.hmojicag.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class HandWrittenDigitsZeroToNineDataSet extends RandomAccessDataset {
    private final List<float[]> X;
    private final List<float[]> Y;

    private HandWrittenDigitsZeroToNineDataSet(Builder builder) {
        super(builder);
        X = builder.X;
        Y = builder.Y;
    }

    @Override
    public Record get(NDManager manager, long index) {
        float[] x = X.get(Math.toIntExact(index));
        float[] y = Y.get(Math.toIntExact(index));
        NDArray datum = manager.create(x);
        NDArray label = manager.create(y);
        return new Record(new NDList(datum), new NDList(label));
    }

    public List<float[]> getX() {
        return X;
    }

    public List<float[]> getY() {
        return Y;
    }

    @Override
    protected long availableSize() {
        return X.size();
    }

    @Override
    public void prepare(Progress progress) {
        // This method is empty on purpose
    }

    public static final class Builder extends BaseBuilder<Builder> {
        List<float[]> X;
        List<float[]> Y;

        @Override
        protected Builder self() {
            return this;
        }

        public HandWrittenDigitsZeroToNineDataSet build() throws IOException {
            // Reads the training set
            ObjectMapper mapper = new ObjectMapper();
            X = mapper.readValue(new File("src/main/resources/W2/X.json"), new TypeReference<List<float[]>>(){});
            Y = mapper.readValue(new File("src/main/resources/W2/y.json"), new TypeReference<List<float[]>>(){});
            return new HandWrittenDigitsZeroToNineDataSet(this);
        }
    }
}
