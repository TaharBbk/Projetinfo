package tests;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import neuronalnetworks.Layer;

public class JUnitLayer {
	
	Layer underTestLayer;
	Layer underTestLayer2;
	
	@Before
	public void setUp() {
		double[] val = {-0.5,0.5,0.8};
		double[][] weight = {{1,0.5,0.2},
							 {0.3,0.3,0},
							 {-0.7,0.6,-0.4}};
		underTestLayer = new Layer(val);
		underTestLayer2 = new Layer(val, weight, underTestLayer);
		
	}
	
	@After
	public void tearDown() {
		underTestLayer = null;
	}
	
	@Test
	public void testActivate() {
		underTestLayer.activate();
		double[] calculatedResult = {-5E-4,
									5E-4,
									8E-4};
		Assert.assertArrayEquals(underTestLayer.getValues(), calculatedResult, 1E-8);	
	}
	
	@Test
	public void testPropagate() {
		underTestLayer2.propagate();
		double[] calculatedResult = {-9.1E-7,3.8E-7,-4.2E-7};
		Assert.assertArrayEquals(underTestLayer.getValues(), calculatedResult, 1E-8);
	}
	
	@Test
	public void testBackPropagation(){
		underTestLayer.backprop_init(new double[] {-1, 1, -1}, 1.0);
		double[][] calculatedWeight = {{1.4686682,-0.0523056,2.172679},
				 			 {-0.1686682,0.8523056,-1.972679},
				 			 {-1.4498692,1.4836890,-3.5562864}};
		for (int i=0; i<3; i++){
			Assert.assertEquals(underTestLayer.getPrecedent(), underTestLayer2);
			Assert.assertArrayEquals(underTestLayer2.getWeights()[i], calculatedWeight[i], 0.00001);
		}
	}
	

}
