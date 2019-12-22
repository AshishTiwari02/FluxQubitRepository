package com.HandleCookies;

import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.openqa.selenium.By;
import org.openqa.selenium.Cookie;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

public class workOnCookies {

	public static void main(String[] args) {
	
		String driverPath = "webdriver.chrome.driver";
		String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";

		System.setProperty(driverPath, driverURL);
		WebDriver driver = new ChromeDriver(); 
	
		String baseUrl = "https://www.amazon.in/";
	   
	    
		try {
			/*
					Cookies Method in Selenium		
			driver.manage().getCookies();
			driver.manage().getCookieNamed("");
			driver.manage().addCookie(null);
			driver.manage().deleteCookie(null);
			driver.manage().deleteAllCookies();
			driver.manage().deleteCookieNamed("Cookie name");
			*/
			 driver.get(baseUrl);
			 
			 // Method one -> getCookies
			 Set<Cookie> cook = driver.manage().getCookies();
			 System.out.println("************* GetCookies ***********************");
			 System.out.println("All Cookies :  " + cook.size());
			 for (Cookie iter : cook)
			 {
				 System.out.println("["+iter.getName()+"] -> [ "+iter.getValue()+"] :-> [ "+iter.getExpiry()+"]");
				 
			 }
			
			// Method Two -> getCookieNamed
			 System.out.println();
			 System.out.println("************* getCookieNamed ***********************");
			 System.out.print("Cookie By Name " + driver.manage().getCookieNamed("session-id-time"));
			 
			// Method Three -> addCookie
			 System.out.println();
			 System.out.println("************* addCookie ***********************");
			 Cookie cObj = new Cookie("myCookie","123456789");
			 driver.manage().addCookie(cObj);
			 
			 Set<Cookie> cooknew = driver.manage().getCookies();
			
			 System.out.println("Added Cookie");
			 System.out.println("All Cookies :  " + cooknew.size());
			 
			// Method four -> DeleteCookieNamed
			 System.out.println();
			 System.out.println("************* deleteCookieNamed ***********************\n");
			 driver.manage().deleteCookieNamed("session-id-time");
			 Set<Cookie> cooknewdel = driver.manage().getCookies();
				
			 System.out.println("After Deletion Cookie");
			 System.out.println("All Cookies :  " + cooknewdel.size());
			 
			// Method Five -> DeleteAllCookies
			 System.out.println("************* deleteAllCookies ***********************\n");
			driver.manage().deleteAllCookies();
			
			// Method Six -> DeleteCookieNamed
			 System.out.println("************* deleteCookieNamed ***********************\n");
			driver.manage().deleteCookie(cObj);
			
			
			 
		}
		
		catch(Exception e) {
			System.out.println("Exception Descrption : " +e);
			}
		finally {
			driver.quit();
		
		
		

	}

}
}