package tests;

import org.junit.Assert;
import org.junit.Test;

import neuralnetworks.Layer;

public class JUnitIntegration {
	
	@Test
	public void TestVectorVector() {
		
		double[] v1 = {0, 1, 2};
		double[] v2 = {3, 5};
		double[][] M1 = {{ 0, 3, 6},
						{0, 5, 10}};
		
		
		Assert.assertArrayEquals(M1[0], Layer.productVectorVector(v1, v2)[0], 1E-8);
		Assert.assertArrayEquals(M1[1], Layer.productVectorVector(v1, v2)[1] ,1E-8);
		
	}
	
	@Test
	public void TestVectorMatrix() {
		
		double[][] M1 = {{ 1, 0 },
						{0, 1}};
		double[] v1 = {0, 1};
		
		Assert.assertArrayEquals(v1, Layer.productMatrixVector(M1, v1), 0.001);
		
	}
	
	@Test
	public void TestMatrixMatrix() {
		
		double[][] M2 = {{ 0, 1 },
						{1, 0}};
		double[][] M1 = {{1, 0}, 
						{0, 1}};
		
		Assert.assertArrayEquals(M2[0], Layer.productMatrixMatrix(M1, M2)[0], 0.001);
		Assert.assertArrayEquals(M2[1], Layer.productMatrixMatrix(M1, M2)[1], 0.001);
		
	}
	
	@Test
	public void TestHadamartProduct() {
		
		double[] v1 = {1, 2, 3};
		double[] v2 = {3, 2, 1};
		double[] v3 = {3, 4, 3};
		
		Assert.assertArrayEquals(v3, Layer.hadamartProduct(v1, v2), 0.001);
		
	}
	
	@Test
	public void TestTranspose() {
		
		double[][] M1 = {{0, 1},
						{1, 0},
						{0,1}};
		double[][] M2 = {{0, 1, 0},
						{1, 0, 1}};
		
		
		Assert.assertArrayEquals(M1[0], Layer.transpose(M2)[0], 1E-8);
		Assert.assertArrayEquals(M1[1], Layer.transpose(M2)[1], 1E-8);
		
	
	}
		
		
	
}
