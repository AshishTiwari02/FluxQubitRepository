package com.TestLearn;

import org.testng.annotations.Test;

public class exceptionTimeOutTest {
	
	@Test(expectedExceptions=NumberFormatException.class)
	public void checkException()
	{
		String x = "101A";
		Integer.parseInt(x);
		
	}
	
	
}
