package io.practice;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.api.TestReporter;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.condition.EnabledOnJre;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.JRE;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.Tag;



@DisplayName("Execution of MainDevCode for Testing")
class MainDevCodesTest {
	MainDevCodes md;
	TestInfo testinfo;
	TestReporter testreporter;
	
	@BeforeAll
	static void beforeEveryInit()
	{
		System.out.println("Setting up the configuration.\n");
	}
	
	@BeforeEach
	void initMethod(TestInfo testinfo,TestReporter testreporter)
	{
		 this.testinfo=testinfo;
		 this.testreporter=testreporter;
		 md = new MainDevCodes();
		 System.out.println("Before Each method for every test initialization.");
	}
	
	@DisplayName("Testing Add Method")
	void testAdd()
	{	
		testreporter.publishEntry("Running " + testinfo.getDisplayName()+" with tags "+ testinfo.getTags());
		int expected = 7;
		int actual  =md.add(2, 5);
		assertEquals(expected, actual,"Both values are diffrent");
		System.out.println("Test Scanerios for Add Method.");
	}

	@DisplayName("Testing Area Method")
	void testArea()
	{
		testreporter.publishEntry("Running " + testinfo.getDisplayName()+" with tags "+ testinfo.getTags());
		double expected = 314.1592653589793;
		double actual  =md.computerArea(10);
		assertEquals(expected, actual,"Area is diffrent than expected");
		System.out.println("Test Scanerios for Area Method.");
	}
	
	@DisplayName("Testing Exception Method")
	@Tag("Math")
	void testDivide()
	{
		testreporter.publishEntry("Running " + testinfo.getDisplayName()+" with tags "+ testinfo.getTags());
		boolean server = true;
		assumeTrue(server);
		assertThrows(ArithmeticException.class , ()-> md.divide(1, 0),"Divide by Zero throw");
		System.out.println("Test Scanerios for Exceptions Method.");
	}
	
	
	@Disabled
	@DisplayName("Testing Disable check method")
	void testDisable()
	{
		
		fail("Scripts needs to designed");
	}
	
	@EnabledOnOs(OS.LINUX)
	@DisplayName("Testing Conditions based")
	void testOnOS()
	{
		fail("Scripts needs to tested in Linux OS");
	}
	
	@EnabledOnJre(JRE.JAVA_11)
	@DisplayName("Testing Conditions based")
	void testOnJRE()
	{
		fail("Scripts needs to tested in JRE-11.");
	}
	
	@DisplayName("Testing Multiply Method")
	void testMultiply()
	{	
		assertAll(
					() -> assertEquals(40, md.multiply(5, 8)),
					() -> assertEquals(-1, md.multiply(1, -1)),
					() -> assertEquals(10, md.multiply(-2, -5))
				);
	}
	
	@Nested
	@DisplayName("Sum Method verification")
	class Sum
		{
			@DisplayName("Testing Positive number sum method")
			void testSumPositive()
			{	
				assertEquals(6, md.sum(1, 2, 3),"Check sum for +ve numbers");
				
			}
		
			@DisplayName("Testing Negative number sum method")
			void testSumNegative()
			{	
				assertEquals(-6, md.sum(-1, -2, -3),"Check sum for -ve numbers");
				
			}
			
			@DisplayName("Testing Positive and Negative number sum method")
			void testSumBoth()
			{	
				assertEquals(20, md.sum(10,-20,30),"Check sum for +ve numbers");
				
			}		
		}

	@DisplayName("Testing Sub Method")
	@RepeatedTest(3)
	@Tag("ReporterTest")
	void testSub()
	{
		testreporter.publishEntry("Running " + testinfo.getDisplayName()+" with tags "+ testinfo.getTags());
		assertEquals(4, md.sub(6, 2),"Check sub for two..");
		System.out.println("Test Scanerios for Sub Method.");
	}
	
	@AfterEach
	void cleanUp()
	{
		System.out.println("Clean up for next Test.....\n");
		
	}
	
}
